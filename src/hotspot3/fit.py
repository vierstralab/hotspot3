import numpy.ma as ma
import numpy as np
from scipy import stats as st
from hotspot3.models import NoContigPresentError, ProcessorConfig, FitResults
from hotspot3.utils import wrap_masked
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


class GlobalBackgroundFit(BackgroundFit):
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
    def fit(self, array: ma.MaskedArray) -> FitResults:
        agg_cutcounts = array.copy()

        per_window_quantiles, per_window_trs, rmsea = self.find_per_window_tr(agg_cutcounts)

        high_signal_mask = (agg_cutcounts >= per_window_trs).filled(False)
        agg_cutcounts[high_signal_mask] = np.nan

        p, r, enough_bg_mask, poisson_fit_params = self.sliding_method_of_moments_fit(agg_cutcounts)

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

        poisson_fit_positions = ma.where(mean >= var)[0]
        if len(poisson_fit_positions) > 0:
            p[poisson_fit_positions] = 0
            r[poisson_fit_positions] = np.inf

            poisson_fit_params = np.empty((len(poisson_fit_positions), 3), dtype=np.float64)
            poisson_fit_params[:, 0] = poisson_fit_positions
            poisson_fit_params[:, 1] = mean[poisson_fit_positions]
            poisson_fit_params[:, 2] = var[poisson_fit_positions]
        else:
            poisson_fit_params = None

        return p, r, enough_bg_mask, poisson_fit_params
    
    @wrap_masked
    def find_per_window_tr(self, agg_cutcounts):
        original_shape = agg_cutcounts.shape
        collapsed_bg_window = self.config.bg_window // self.config.window
        min_count = self.config.min_mappable_bg // self.config.window_stats_step
        collapsed_agg_cutcounts = rolling_view_with_nan_padding_subsample(
            agg_cutcounts[::self.config.window],
            window_size=collapsed_bg_window,
            subsample_step=self.config.window_stats_step
        ) # Shape (n_points, collapsed_bg_window)

        right_edges = np.round(
            np.nanquantile(collapsed_agg_cutcounts, self.get_all_quantiles(), axis=1).T
        ).astype(np.int32) # Shape (n_points, num_bins)

        # collapsed_agg_cutcounts = collapsed_agg_cutcounts[collapsed_agg_cutcounts < right_edges[:, -1][:, None]]
        
        value_counts = np.full_like(right_edges, 0, dtype=np.int16)

        left_edges = np.zeros(right_edges.shape[0])
        for i in range(right_edges.shape[1]):
            bin_membership = (collapsed_agg_cutcounts[:, i] >= left_edges) & (collapsed_agg_cutcounts[:, i] < right_edges[:, i])
            value_counts[:, i] = np.sum(bin_membership, axis=1)
            left_edges = right_edges[:, i]


        enough_bg_mask = (~np.isnan(collapsed_agg_cutcounts)).sum(axis=1) > min_count

        mean = np.nanmean(collapsed_agg_cutcounts, axis=1)
        var = np.nanvar(collapsed_agg_cutcounts, axis=1, ddof=1)

        p = self.p_from_mean_and_var(mean, var)
        r = self.r_from_mean_and_var(mean, var)

        poisson_indices = np.where(mean >= var)[0]
        p[poisson_indices] = 0
        r[poisson_indices] = np.inf

        rmsea = np.full(mean.shape, np.nan)

        negbin_fit = enough_bg_mask.copy()
        negbin_fit[poisson_indices] = False

        rmsea[negbin_fit] = self.calc_rmsea_all_windows(
            r[negbin_fit], p[negbin_fit],
            right_edges.T[negbin_fit],
            value_counts.T[negbin_fit]
        )

        rmsea[poisson_indices] = -1

        bad_fits = np.where(enough_bg_mask & (rmsea > 0.05))[0]
        trs = np.full_like(rmsea, np.nan)
        trs[~bad_fits] = right_edges[~bad_fits, -1]

        for i, r_edge in enumerate(right_edges[:, ::-1].T):
            if bad_fits.size == 0 or i == 20:
                break
            collapsed_agg_cutcounts = collapsed_agg_cutcounts[collapsed_agg_cutcounts <= r_edge[:, None]]
            right_edges = right_edges[:, :-1]
            value_counts = value_counts[:, :-1]

            mean = np.nanmean(collapsed_agg_cutcounts, axis=1)
            var = np.nanvar(collapsed_agg_cutcounts, axis=1, ddof=1)

            enough_bg_mask = (~np.isnan(collapsed_agg_cutcounts)).sum(axis=1) > min_count

            p = self.p_from_mean_and_var(mean, var)
            r = self.r_from_mean_and_var(mean, var)

            poisson_indices = np.where(mean >= var)[0]
            p[poisson_indices] = 0
            r[poisson_indices] = np.inf

            negbin_fit = enough_bg_mask.copy()
            negbin_fit[poisson_indices] = False

            rmsea[negbin_fit] = self.calc_rmsea_all_windows(
                r[negbin_fit], p[negbin_fit],
                right_edges.T[negbin_fit],
                value_counts.T[negbin_fit]
            )

            rmsea[poisson_indices] = -1

            bad_fits = np.where(enough_bg_mask & (rmsea > 0.05))[0]
            new_good_fits = np.isnan(trs) & ~bad_fits
            trs[new_good_fits] = right_edges[new_good_fits, -1]

        result = np.full_like(original_shape, np.nan)
        result[::self.config.window][::self.config.window_stats_step] = trs
        return result

    def get_observed_per_bin(self, cutcounts, cutcounts_bin_edges, collapsed_bg_window):
        observed_per_bin = np.zeros_like(cutcounts_bin_edges)
        collapsed_bg_window = np.ceil(self.config.bg_window / self.config.window_stats_step).astype(int)

        for i in range(cutcounts_bin_edges.shape[0] - 1):
            left, right = cutcounts_bin_edges[i], cutcounts_bin_edges[i + 1]
            observed_per_bin[i] = self.centered_running_nansum(
                (cutcounts >= left) & (cutcounts < right),
                collapsed_bg_window
            )

        return observed_per_bin

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
    def move_sum_with_dtype(self, array, window, min_count):
        array = np.asarray(array, np.float32)
        return bn.move_sum(array, window, min_count=min_count).astype(np.float32)
    
    @wrap_masked
    def move_rank_with_dtype(self, array, window, min_count):
        array = np.asarray(array, np.float32)
        return (bn.move_rank(array, window, min_count=min_count).astype(np.float32) + 1.) / 2.
    
    @wrap_masked
    def centered_running_nanvar(self, array, window, min_count):
        return correct_offset(bn.move_var, array, window, ddof=1, min_count=min_count).astype(np.float32)

    @wrap_masked
    def centered_running_nanmean(self, array, window, min_count):
        return correct_offset(bn.move_mean, array, window, min_count=min_count).astype(np.float32)
    
    @wrap_masked
    def centered_running_nansum(self, array: np.ndarray, window, min_count):
        return correct_offset(self.move_sum_with_dtype, array, window, min_count=min_count)
    
    @wrap_masked
    def stratify(self, array, step):
        res = np.full_like(array, np.nan)
        res[::step] = array[::step]
        return res
    
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

    @wrap_masked
    def centered_running_rank(self, array, window, min_count):
        """
        Calculate the rank of the center value in a moving window.
        Ignore NaN values.
        """
        assert window % 2 == 1, "Window size should be odd"
        half_window = window // 2 + 1
        not_nan = (~np.isnan(array)).astype(np.float32)

        trailing_ranks = self.move_rank_with_dtype(array, half_window, min_count=min_count)
        trailing_ranks *= self.move_sum_with_dtype(not_nan, half_window, min_count) - 1

        leading_ranks = self.move_rank_with_dtype(array[::-1], half_window, min_count=min_count)[::-1]
        leading_ranks *= self.move_sum_with_dtype(not_nan[::-1], half_window, min_count)[::-1] - 1
        
        combined_ranks = trailing_ranks + leading_ranks
        denom = self.centered_running_nansum(not_nan, window, min_count=min_count) - 1
        denom[denom == 0] = 1.
        combined_ranks /= denom
        return combined_ranks

    @wrap_masked
    def calc_rmsea_all_windows(
        self,
        r, p,
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


def correct_offset(function, array, window, *args, **kwargs):
    """
    Correct offset of a trailing running window to make it centered.
    """
    assert window % 2 == 1, "Window size should be odd"
    offset = window // 2
    result = function(
        np.pad(array, (0, offset), mode='constant', constant_values=np.nan),
        window,
        *args,
        **kwargs
    )
    return result[offset:]


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
    
    return subsampled_view
