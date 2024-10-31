import numpy.ma as ma
import numpy as np
from scipy import stats as st
from hotspot3.models import NoContigPresentError, ProcessorConfig, FitResults
from hotspot3.utils import wrap_masked, correct_offset
from hotspot3.logging import setup_logger
import bottleneck as bn


class BackgroundFit:
    def __init__(self, config: ProcessorConfig=None, logger=None, name=None):
        if config is None:
            config = ProcessorConfig()
        self.config = config

        if logger is None:
            logger = setup_logger()
        self.logger = logger
        if name is None:
            name = self.__class__.__name__
        self.name = name


    def fit(self) -> FitResults:
        raise NotImplementedError
    
    @wrap_masked
    def get_mean_and_var(self, array: np.ndarray):
        mean = np.nanmean(array, axis=0, dtype=np.float32)
        var = np.nanvar(array, ddof=1, axis=0, dtype=np.float32)
        return mean, var

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
    
    def get_min_bg_tr(self, array):
        return np.nanquantile(array, self.config.min_background_prop, axis=0)
    
    def get_max_bg_tr(self, array: np.ndarray):
        return np.nanquantile(array, self.config.max_background_prop, axis=0)
    
    def get_signal_bins(self, array: np.ndarray, min_bg_tr=None):
        if min_bg_tr is None:
            min_bg_tr = self.get_min_bg_tr(array)
        max_bg_tr = self.get_max_bg_tr(array)
        n_signal_bins = min(np.nanmax(max_bg_tr - min_bg_tr), self.config.num_signal_bins)
        n_signal_bins = round(n_signal_bins)
        return np.round(np.linspace(
            min_bg_tr,
            max_bg_tr,
            n_signal_bins + 1
        )).astype(np.int32)
        
    
    def get_all_bins(self, array: np.ndarray):
        min_bg_tr = self.get_min_bg_tr(array)
        n_bg_bins = min(np.nanmax(min_bg_tr), self.config.num_background_bins)
        n_bg_bins = round(n_bg_bins)
        bg_bins = np.round(np.linspace(
            0,
            min_bg_tr,
            n_bg_bins + 1
        )).astype(np.int32)
        signal_bins = self.get_signal_bins(array, min_bg_tr=min_bg_tr)
        return np.concatenate([bg_bins[:-1], signal_bins])

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
        trs = self.get_signal_bins(agg_cutcounts)[::-1]
        for tr in trs:
            p, r = self.fit_for_tr(agg_cutcounts, tr)
            rmsea = self.calc_rmsea_for_tr(counts, unique, p, r, tr)
            res.append((tr, rmsea))
            if rmsea <= self.config.rmsea_tr:
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
            self.logger.warning(f"{self.name}: Not enough background to fit the global mean. Skipping.")
            raise NoContigPresentError
        
        mean, variance = self.get_mean_and_var(agg_cutcounts)
        return mean, variance

    def calc_rmsea_for_tr(self, obs, unique_cutcounts, p, r, tr):
        mask = unique_cutcounts < tr
        unique_cutcounts = unique_cutcounts[mask]
        obs = obs[mask].copy()
        N = sum(obs)
        exp = st.nbinom.pmf(unique_cutcounts, r, 1 - p) / st.nbinom.cdf(tr - 1, r, 1 - p) * N
        # chisq = sum((obs - exp) ** 2 / exp)
        G_sq = np.sum(calc_g_sq(obs, exp))
        # obs[obs == 0] = 1
        # G_sq = 2 * sum(obs * np.log(obs / exp))
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

    def __init__(self, config=None, logger=None, name=None):
        super().__init__(config, logger=logger, name=name)
        self.sampling_step = self.config.bg_window // self.config.signal_prop_n_samples
        self.interpolation_step = self.config.signal_prop_step // self.sampling_step
    
    def value_counts_per_bin(self, strided_cutcounts, bin_edges):
        left_edges = bin_edges[0]
        right_edges = bin_edges[1:]
        value_counts = np.full_like(right_edges, 0, dtype=np.int32)
        for i in range(right_edges.shape[0]):
            bin_membership = (strided_cutcounts >= left_edges[None, :]) & (strided_cutcounts < right_edges[i][None, :])
            value_counts[i] = np.sum(bin_membership, axis=0)
            left_edges = right_edges[i]
        
        return value_counts

    def fit_for_bin(self, collapsed_agg_cutcounts):
        min_count = round(self.config.signal_prop_n_samples * self.config.min_mappable_bg / self.config.bg_window)
        enough_bg_mask = np.sum(~np.isnan(collapsed_agg_cutcounts), axis=0) > min_count
        mean = np.full(collapsed_agg_cutcounts.shape[1], np.nan, dtype=np.float32)
        var = np.full(collapsed_agg_cutcounts.shape[1], np.nan, dtype=np.float32)

        mean[enough_bg_mask] = np.nanmean(collapsed_agg_cutcounts[:, enough_bg_mask], axis=0)
        var[enough_bg_mask] = np.nanvar(collapsed_agg_cutcounts[:, enough_bg_mask], axis=0, ddof=1)

        p = self.p_from_mean_and_var(mean, var)
        r = self.r_from_mean_and_var(mean, var)
        poisson_params = self.pack_poisson_params(mean, var, p, r)

        return p, r, enough_bg_mask, poisson_params
    
    def wrap_rmsea_valid_fits(self, p, r, bin_edges, value_counts, enough_bg_mask=None, poisson_params=None):
        result = np.full_like(p, np.nan)
        if enough_bg_mask is None:
            enough_bg_mask = np.ones_like(p, dtype=bool)
        if poisson_params is None:
            poisson_params = np.zeros((0, 3), dtype=np.float64)

        indices = poisson_params.astype(np.int32)[:, 0]
        if len(indices) > 0:
            result[indices] = self.calc_rmsea_all_windows(
                st.poisson(poisson_params[:, 1]),
                n_params=1,
                bin_edges=bin_edges[:, indices],
                value_counts_per_bin=value_counts[:, indices]
            )

        negbin_fit_mask = enough_bg_mask.copy()
        negbin_fit_mask[indices] = False

        result[negbin_fit_mask] = self.calc_rmsea_all_windows(
            st.nbinom(r[negbin_fit_mask], 1 - p[negbin_fit_mask]),
            n_params=2,
            bin_edges=bin_edges[:, negbin_fit_mask],
            value_counts_per_bin=value_counts[:, negbin_fit_mask]
        )
        return result
        
    @wrap_masked
    def find_per_window_tr(self, array: ma.MaskedArray):
        original_shape = array.shape
        agg_cutcounts = array[::self.sampling_step].copy()

        strided_agg_cutcounts = rolling_view_with_nan_padding_subsample(
            agg_cutcounts,
            window_size=self.config.signal_prop_n_samples,
            subsample_step=self.interpolation_step
        ) # shape (bg_window, n_points)
        bin_edges = self.get_all_bins(strided_agg_cutcounts)
        value_counts = self.value_counts_per_bin(strided_agg_cutcounts, bin_edges)

        best_tr = bin_edges[-1].copy()
        remaing_fits_mask = np.ones_like(best_tr, dtype=bool)
        best_rmsea = np.full_like(best_tr, np.inf, dtype=np.float32)
        step = 1

        for i in range(0, self.config.num_signal_bins, step):
            if remaing_fits_mask.sum() == 0:
                break
            if i != 0:
                bin_edges = bin_edges[:-step, :]
                value_counts = value_counts[:-step, :]
            strided_agg_cutcounts[strided_agg_cutcounts >= bin_edges[-1, :]] = np.nan

            edges = bin_edges[:, remaing_fits_mask]
            counts = value_counts[:, remaing_fits_mask]

            p, r, enough_bg_mask, poisson_params = self.fit_for_bin(
                strided_agg_cutcounts[:, remaing_fits_mask]
            )

            rmsea = self.wrap_rmsea_valid_fits(p, r, edges, counts, enough_bg_mask, poisson_params) # shape of r
            
            successful_fits = enough_bg_mask & (rmsea <= self.config.rmsea_tr)
            # best fit found
            better_fit = np.where(
                (rmsea < best_rmsea[remaing_fits_mask]),
                True,
                False
            )

            best_tr[remaing_fits_mask] = np.where(
                better_fit,
                edges[-1],
                best_tr[remaing_fits_mask]
            )
            best_rmsea[remaing_fits_mask] = np.where(
                better_fit,
                rmsea,
                best_rmsea[remaing_fits_mask]
            )

            # Not enough background
            best_tr[remaing_fits_mask] = np.where(
                ~enough_bg_mask,
                np.nan,
                best_tr[remaing_fits_mask]
            )
            best_rmsea[remaing_fits_mask] = np.where(
                ~enough_bg_mask,
                np.nan,
                best_rmsea[remaing_fits_mask]
            )

            remaing_fits_mask[remaing_fits_mask] = ~successful_fits | ~enough_bg_mask 
            self.logger.debug(f"{self.name}: Remaining fits: {remaing_fits_mask.sum()}")

        subsampled_indices = np.arange(
            0, original_shape[0], self.sampling_step, dtype=np.uint32
        )[::self.interpolation_step]

        return self.interpolate_nan(original_shape[0], best_tr, subsampled_indices), self.interpolate_nan(original_shape[0], best_rmsea, subsampled_indices)


    @wrap_masked
    def calc_rmsea_all_windows(
        self,
        dist: st.rv_discrete,
        n_params: int,
        bin_edges, # right edges of the bins, left edge of the first bin is 0, right edge not inclusive
        value_counts_per_bin, # number of observed cutcounts in each bin for each window
    ):
        """
        Calculate RMSEA for all sliding windows.
        """
        bg_sum_mappable = np.sum(value_counts_per_bin, axis=0)
        # print(bin_edges)
        sf_values = dist.sf(bin_edges - 1)
        sf_diffs = -np.diff(sf_values, axis=0)
        assert sf_diffs.shape == value_counts_per_bin.shape, "SF diffs shape should match value counts shape"
        norm_coef = 1 - sf_values[-1]
        expected_counts = (sf_diffs * bg_sum_mappable / norm_coef)
        G_sq = np.sum(
            calc_g_sq(value_counts_per_bin, expected_counts), 
            axis=0
        )
        #print(list(zip(value_counts_per_bin[:, 0], np.diff(bin_edges[:, 0]), ((value_counts_per_bin > 0) & (np.diff(bin_edges, axis=0) != 0))[:, 0])))
        df = np.sum(
            (value_counts_per_bin > 0) & (np.diff(bin_edges, axis=0) != 0),
            axis=0
        ) - n_params

        rmsea = np.where(df >= 3, np.sqrt(np.maximum(G_sq / df - 1, 0) / (bg_sum_mappable - 1)), -2)
        assert np.sum(np.isnan(rmsea)) == 0, "RMSEA should not contain NaNs"
        return rmsea

    @wrap_masked
    def interpolate_nan(self, original_length, subsampled_arr, subsampled_indices):
        # indices = np.arange(0, len(arr), step, dtype=np.uint32)
        # subsampled_arr = arr[::step].copy()
        
        nan_mask = np.isnan(subsampled_arr)
        subsampled_arr = subsampled_arr[~nan_mask]
        subsampled_indices = subsampled_indices[~nan_mask]

        return np.interp(
            np.arange(original_length, dtype=np.uint32),
            subsampled_indices,
            subsampled_arr,
            left=None,
            right=None,
        )


def calc_g_sq(obs, exp):
    ratio = np.where((exp != 0) & (obs != 0), obs / exp, 1)
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
 
    return subsampled_view.T.copy()
