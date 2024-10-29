import numpy.ma as ma
import numpy as np
from scipy import stats as st
from hotspot3.models import NoContigPresentError, ProcessorConfig, FitResults
from hotspot3.utils import wrap_masked, correct_offset
import bottleneck as bn
from scipy.special import betainc


class BackgroundFit:
    def __init__(self, config: ProcessorConfig=None):
        if config is None:
            config = ProcessorConfig()
        self.config = config

    def fit(self) -> FitResults:
        raise NotImplementedError

    @wrap_masked
    def p_from_mean_and_var(self, mean: np.ndarray, var: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            p = np.array(1 - mean / var, dtype=np.float32)
        return p
    
    @wrap_masked
    def r_from_mean_and_var(self, mean: np.ndarray, var: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.array(mean ** 2 / (var - mean), dtype=np.float32)
        return r
    
    def get_signal_quantiles(self, n_bins=20):
        return np.linspace(self.config.min_background_prop, self.config.max_background_prop, n_bins + 1)
    
    def get_all_quantiles(self, signal_bins=20, background_bins=10):
        background_bin_edges = np.linspace(0, self.config.min_background_prop, background_bins + 1)[1:-1]
        signal_bin_edges = self.get_signal_quantiles(signal_bins)
        return np.concatenate([background_bin_edges, signal_bin_edges])

    def pack_poisson_params(self, mean, var, p, r):
        poisson_fit_positions = np.where(mean >= var)[0]
        poisson_fit_params = np.empty((len(poisson_fit_positions), 3), dtype=np.float64)
        poisson_fit_params[:, 0] = poisson_fit_positions
        poisson_fit_params[:, 1] = mean[poisson_fit_positions]
        poisson_fit_params[:, 2] = var[poisson_fit_positions]

        p[poisson_fit_positions] = 0
        r[poisson_fit_positions] = np.inf
        return poisson_fit_params


class GlobalBackgroundFit(BackgroundFit):
    """
    Class to fit the background distribution globally (for chromosome)
    """
    def fit(self, array: ma.MaskedArray) -> FitResults:
        agg_cutcounts = ma.masked_invalid(array).compressed()
        unique, counts = self.hist_data_for_tr(agg_cutcounts)

        res = []
        quantiles = self.get_signal_quantiles()
        trs = np.unique(np.quantile(agg_cutcounts, quantiles))[::-1]
        for tr in trs:
            p, r = self.fit_for_tr(agg_cutcounts, tr)
            rmsea = self.calc_rmsea_for_tr(counts, unique, p, r, tr)
            res.append((tr, rmsea))
            if rmsea <= 0.05:
                break
        else:
            tr, rmsea = min(res, key=lambda x: x[1])
        quantile = np.sum(agg_cutcounts < tr) / agg_cutcounts.shape[0]

        return FitResults(p.squeeze(), r.squeeze(), rmsea.squeeze(), quantile, tr)

    def fit_for_tr(self, agg_cutcounts, tr):
        agg_cutcounts = agg_cutcounts[agg_cutcounts < tr]
        mean, var = self.estimate_global_mean_and_var(agg_cutcounts)

        p = self.p_from_mean_and_var(mean, var)
        r = self.r_from_mean_and_var(mean, var)
        return p, r
    

    def hist_data_for_tr(self, agg_cutcounts, tr=None):
        if tr is not None:
            agg_cutcounts = agg_cutcounts[agg_cutcounts < tr]
        unique, counts = np.unique(agg_cutcounts, return_counts=True)
        return unique, counts


    def estimate_global_mean_and_var(self, agg_cutcounts: np.ndarray):
        has_enough_background = np.count_nonzero(agg_cutcounts) / agg_cutcounts.size > self.config.nonzero_windows_to_fit
        if not has_enough_background:
            raise NoContigPresentError
        
        mean = np.mean(agg_cutcounts)
        variance = np.var(agg_cutcounts, ddof=1)
        return mean, variance

    def calc_rmsea_for_tr(self, obs, unique_cutcounts, p, r, tr):
        mask = unique_cutcounts < tr
        unique_cutcounts = unique_cutcounts[mask]
        obs = obs[mask].copy()
        N = sum(obs)
        exp = st.nbinom.pmf(unique_cutcounts, r, 1 - p) / st.nbinom.cdf(tr - 1, r, 1 - p) * N
        # chisq = sum((obs - exp) ** 2 / exp)
        obs[obs == 0] = 1
        G_sq = 2 * sum(obs * np.log(obs / exp))
        df = len(obs) - 2
        return np.sqrt(np.maximum(G_sq / df - 1, 0) / (N - 1))


class WindowBackgroundFit(BackgroundFit):
    """
    Class to fit the background distribution in a running window fashion
    """
    def fit(self, array: ma.MaskedArray, per_window_trs) -> FitResults:
        agg_cutcounts = array.copy()

        high_signal_mask = (agg_cutcounts >= per_window_trs).filled(False)
        agg_cutcounts[high_signal_mask] = np.nan

        p, r, enough_bg_mask, poisson_fit_params = self.sliding_method_of_moments_fit(agg_cutcounts)

        rmsea = np.full_like(p, np.nan)
        per_window_quantiles = np.full_like(p, np.nan)
        return FitResults(
            p, r, rmsea,
            fit_quantile=per_window_quantiles,
            fit_threshold=per_window_trs,
            enough_bg_mask=enough_bg_mask,
            poisson_fit_params=poisson_fit_params
        )
    
    def sliding_method_of_moments_fit(self, agg_cutcounts: ma.MaskedArray, min_count=None, window=None):
        mean, var = self.sliding_mean_and_variance(agg_cutcounts, min_count=min_count, window=window)
        enough_bg_mask = ~mean.mask

        p = self.p_from_mean_and_var(mean, var).filled(np.nan)
        r = self.r_from_mean_and_var(mean, var).filled(np.nan)

        poisson_fit_params = self.pack_poisson_params(mean, var, p, r)
        return p, r, enough_bg_mask, poisson_fit_params

    def sliding_mean_and_variance(self, array: ma.MaskedArray, min_count=None, window=None):
        if window is None:
            window = self.config.bg_window

        if min_count is None:
            min_count = self.config.min_mappable_bg

        mean = self.centered_running_nanmean(array, window, min_count=min_count)
        var = self.centered_running_nanvar(array, window, min_count=min_count)
        mean = ma.masked_invalid(mean)
        var = ma.masked_invalid(var)
        return mean, var
    
    @wrap_masked
    @correct_offset
    def centered_running_nansum(self, array: np.ndarray, window, min_count):
        return bn.move_sum(array, window, min_count=min_count).astype(np.float32)

    @wrap_masked
    @correct_offset
    def centered_running_nanvar(self, array, window, min_count):
        return bn.move_var(array, window, ddof=1, min_count=min_count).astype(np.float32)

    @wrap_masked
    @correct_offset
    def centered_running_nanmean(self, array, window, min_count):
        return bn.move_mean(array, window, min_count=min_count).astype(np.float32)
    
    @wrap_masked
    def stratify(self, array, step):
        res = np.full_like(array, np.nan)
        res[::step] = array[::step]
        return res


class StridedFit(BackgroundFit):
    """
    Class to fit the background distribution using a strided approach.
    Extemely computationally taxing, use large stride values to compensate.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.bg_window = self.config.bg_window // self.config.window
        self.min_count = self.config.min_mappable_bg // self.config.window_stats_step
        self.window = self.config.window
        self.window_stats_step = self.config.window_stats_step


    def get_right_edges(self, strided_agg_cutcounts):
        return np.round(
            np.nanquantile(
                strided_agg_cutcounts,
                self.get_all_quantiles(),
                axis=0
            )
        ).astype(np.int32)
    
    def value_counts_per_bin(self, strided_cutcounts, right_edges):
        value_counts = np.full_like(right_edges, 0, dtype=np.int16)

        left_edges = np.zeros(right_edges.shape[0])
        for i in range(right_edges.shape[1]):
            bin_membership = (strided_cutcounts[:, i] >= left_edges) & (strided_cutcounts[:, i] < right_edges[:, i])
            value_counts[:, i] = np.sum(bin_membership, axis=1)
            left_edges = right_edges[:, i]
        
        return value_counts

    def fit_for_bin(self, collapsed_agg_cutcounts):
        enough_bg_mask = np.sum(~np.isnan(collapsed_agg_cutcounts), axis=1) > self.min_count

        mean = np.nanmean(collapsed_agg_cutcounts, axis=1)
        mean[~enough_bg_mask] = np.nan
        var = np.nanvar(collapsed_agg_cutcounts, axis=1, ddof=1)
        var[~enough_bg_mask] = np.nan

        p = self.p_from_mean_and_var(mean, var)
        r = self.r_from_mean_and_var(mean, var)
        poisson_params = self.pack_poisson_params(mean, var, p, r)

        return p, r, enough_bg_mask, poisson_params
    
    def wrap_rmsea_valid_fits(self, p, r, enough_bg_mask, poisson_params, right_edges, value_counts):
        result = np.full_like(p, np.nan)
        indices = poisson_params.astype(np.int32)[:, 0]
        result[indices] = -1

        negbin_fit_mask = enough_bg_mask.copy()
        negbin_fit_mask[indices] = False

        result[negbin_fit_mask] = self.calc_rmsea_all_windows(
            r[negbin_fit_mask], p[negbin_fit_mask],
            right_edges[negbin_fit_mask, :],
            value_counts[negbin_fit_mask, :]
        )
        return result
        
    @wrap_masked
    def find_per_window_tr(self, array: ma.MaskedArray):
        original_shape = array.shape
        agg_cutcounts = array[::self.window].copy()

        strided_agg_cutcounts = rolling_view_with_nan_padding_subsample(
            agg_cutcounts,
            window_size= self.bg_window,
            subsample_step=self.window_stats_step
        ) # shape (bg_window, n_points)

        signal_quantiles = self.get_signal_quantiles()

        right_edges = self.get_right_edges(strided_agg_cutcounts)
        value_counts = self.value_counts_per_bin(strided_agg_cutcounts, right_edges)

        best_tr = np.full_like(right_edges.shape[1], np.nan)
        remaing_fits_mask = np.ones_like(best_tr, dtype=bool)
        for i in range(len(signal_quantiles)):
            if remaing_fits_mask.sum() == 0:
                break
            right_edges = right_edges[:-1, :]
            value_counts = value_counts[:-1, :]
            strided_agg_cutcounts[strided_agg_cutcounts >= right_edges[-1, :]] = np.nan

            edges = right_edges[:, remaing_fits_mask]
            counts = value_counts[:, remaing_fits_mask]

            p, r, enough_bg_mask, poisson_params = self.fit_for_bin(
                strided_agg_cutcounts[:, remaing_fits_mask]
            )

            rmsea = self.wrap_rmsea_valid_fits(p, r, enough_bg_mask, poisson_params, edges, counts) # shape of r
            
            successful_fits = ~enough_bg_mask | (rmsea <= 0.05)
            remaing_fits_mask[remaing_fits_mask] = ~successful_fits
    
            best_tr[remaing_fits_mask] = edges[-1, successful_fits]
        
        result = np.full_like(original_shape, np.nan)
        result[::self.window][::self.window_stats_step] = best_tr
        return self.interpolate_nan(result)


    @wrap_masked
    def calc_rmsea_all_windows(
        self,
        p, r,
        right_bin_edges, # right edges of the bins, left edge of the first bin is 0, right edge not inclusive
        value_counts_per_bin, # number of observed cutcounts in each bin for each window
    ):
        """
        Calculate RMSEA for all sliding windows.
        """
        # assert (bg_window - 1) % step == 0, "Stride should be a divisor of the window size."
        bg_sum_mappable = np.sum(value_counts_per_bin, axis=0)
        tr = right_bin_edges[-1]

        assert len(r) == len(p) == len(bg_sum_mappable) == len(tr)

        G_sq = np.zeros(value_counts_per_bin.shape[1], dtype=np.float32)
        num_nonzero_bins = np.zeros_like(G_sq)
        prev_cdf = 0.0
        norm_coef = betainc(r, tr - 1, 1 - p, dtype=np.float32)
        for i in range(len(right_bin_edges)):
            right = right_bin_edges[i]
            if i == 0:
                nonzero_expected_count_indicator = np.full_like(right, True, dtype=bool)
            else:
                nonzero_expected_count_indicator = right != right_bin_edges[i - 1]

            num_nonzero_bins += nonzero_expected_count_indicator
            observed = value_counts_per_bin[i]

            if i == len(right_bin_edges) - 1:
                new_cdf = norm_coef
            else:
                new_cdf = betainc(r, right - 1, 1 - p, dtype=np.float32)
            expected = (new_cdf - prev_cdf) * bg_sum_mappable / norm_coef
            prev_cdf = new_cdf

            G_sq[nonzero_expected_count_indicator] += calc_g_sq(
                observed[nonzero_expected_count_indicator],
                expected[nonzero_expected_count_indicator],
            )

        df = num_nonzero_bins - 2 # number of bins - number of parameters
        return np.sqrt(np.maximum(G_sq / df - 1, 0) / (bg_sum_mappable - 1))

    @wrap_masked
    def interpolate_nan(self, arr, step):
        indices = np.arange(0, len(arr), step, dtype=np.uint32)
        subsampled_arr = arr[::step].copy()
        
        nan_mask = np.isnan(subsampled_arr)
        if nan_mask.any():
            valid_indices = np.where(~nan_mask)[0]
            nan_indices = np.where(nan_mask)[0]
            subsampled_arr[nan_mask] = np.interp(
                nan_indices, valid_indices, subsampled_arr[valid_indices]
            )

        return np.interp(
            np.arange(len(arr), dtype=np.uint32),
            indices,
            subsampled_arr,
            left=None,
            right=None,
        )


def calc_g_sq(obs, exp):
    ratio = obs / exp
    ratio[ratio == 0] = 1
    return obs * np.log(ratio) * 2


def rolling_view_with_nan_padding_subsample(arr, window_size=501, subsample_step=1000):
    """
    Creates a 2D array where each row is a shifted view of the original array with NaN padding.
    Only every 'subsample_step' row is taken to reduce memory usage.

    Parameters:
        arr (np.ndarray): Input array of shape (n,)
        window_size (int): The total size of the shift (default is 501 for shifts from -250 to +250)
        subsample_step (int): Step size for subsampling (default is 1000)

    Returns:
        np.ndarray: A subsampled array with NaN padding, of shape (n // subsample_step, window_size)
    """
    n = arr.shape[0]
    assert window_size % 2 == 1, "Window size must be odd to have a center shift of 0."
    
    # Calculate padding for out-of-bound shifts
    pad_width = (window_size - 1) // 2
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
    
    # Calculate the number of rows after subsampling
    subsample_count = (n + subsample_step - 1) // subsample_step
    
    # Create a strided view with only every 'subsample_step' row
    shape = (subsample_count, window_size)
    strides = (padded_arr.strides[0] * subsample_step, padded_arr.strides[0])
    subsampled_view = np.lib.stride_tricks.as_strided(padded_arr, shape=shape, strides=strides)
    
    return subsampled_view.T
