import numpy.ma as ma
import numpy as np
from scipy import stats as st
from scipy.special import betainc
from hotspot3.models import NoContigPresentError, ProcessorConfig, FitResults
from hotspot3.utils import wrap_masked, correct_offset, rolling_view_with_nan_padding
from hotspot3.logging import setup_logger
from hotspot3.stats import calc_g_sq, calc_chisq
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

        self.min_mappable_bg = round(self.config.min_mappable_bg_frac * self.config.bg_window)
        self.sampling_step = self.config.signal_prop_sampling_step
        self.interpolation_step = self.config.signal_prop_interpolation_step // self.sampling_step
        self.points_in_bg_window = (self.config.bg_window - 1) // self.sampling_step
        if self.points_in_bg_window % 2 == 0:
            self.points_in_bg_window += 1

    def fit(self) -> FitResults:
        raise NotImplementedError
    
    @wrap_masked
    def get_mean_and_var(self, array: np.ndarray, **kwargs):
        mean = np.nanmean(array, axis=0, dtype=np.float32, **kwargs)
        var = np.nanvar(array, ddof=1, axis=0, dtype=np.float32, **kwargs)
        return mean, var

    @wrap_masked
    def p_from_mean_and_var(self, mean: np.ndarray, var: np.ndarray, where=None):
        with np.errstate(divide='ignore', invalid='ignore', all='ignore'):
            p = np.array(1 - mean / var, dtype=np.float32)
        return p
    
    @wrap_masked
    def r_from_mean_and_var(self, mean: np.ndarray, var: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore', all='ignore'):
            r = np.array(mean ** 2 / (var - mean), dtype=np.float32)
        return r
    
    def get_bg_tr(self, array: np.ndarray, quantile: float):
        all_nan = np.all(np.isnan(array), axis=0)
        if array.ndim == 1:
            return np.nan if all_nan else np.nanquantile(array, quantile)
        result = np.full(array.shape[1], np.nan, dtype=np.float32)
        result[~all_nan] = np.nanquantile(array[:, ~all_nan], quantile, axis=0)
        return result
    
    def get_signal_bins(self, array: np.ndarray, min_bg_tr=None):
        if min_bg_tr is None:
            min_bg_tr = self.get_bg_tr(array, self.config.min_background_prop)
        max_bg_tr = self.get_bg_tr(array, self.config.max_background_prop)
        n_signal_bins = min(np.nanmax(max_bg_tr - min_bg_tr), self.config.num_signal_bins)
        n_signal_bins = round(n_signal_bins)
        
        result = np.full((n_signal_bins + 1, *min_bg_tr.shape), np.nan, dtype=np.float32)
        nan_tr = np.isnan(min_bg_tr) | np.isnan(max_bg_tr)
        result[:, ~nan_tr] = np.round(
            np.linspace(min_bg_tr[~nan_tr], max_bg_tr[~nan_tr], n_signal_bins + 1)
        )
        return result, n_signal_bins
        
    def get_all_bins(self, array: np.ndarray):
        min_bg_tr = self.get_bg_tr(array, self.config.min_background_prop)
        signal_bins, n_signal_bins = self.get_signal_bins(array, min_bg_tr=min_bg_tr)
        n_bg_bins = min(np.nanmax(min_bg_tr), self.config.num_background_bins)
        n_bg_bins = round(n_bg_bins)

        bg_bins = np.full((n_bg_bins + 1, *min_bg_tr.shape), np.nan)
        bg_bins[:, ~np.isnan(min_bg_tr)] = np.round(
            np.linspace(
                0,
                min_bg_tr[~np.isnan(min_bg_tr)],
                n_bg_bins + 1,
            )
        )
        
        return np.concatenate([bg_bins[:-1], signal_bins]), n_signal_bins


class GlobalBackgroundFit(BackgroundFit):
    """
    Class to fit the background distribution globally (for chromosome)
    """
    def fit(self, array: ma.MaskedArray) -> FitResults:
        agg_cutcounts = ma.masked_invalid(array)[::self.config.window].compressed()
        unique, counts = self.hist_data_for_tr(agg_cutcounts)

        res = []
        trs, _ = self.get_signal_bins(agg_cutcounts)
        print(trs)
        for tr in trs:
            p, r = self.fit_for_tr(agg_cutcounts, tr)
            rmsea = self.calc_rmsea_for_tr(counts, unique, p, r, tr)
            res.append((tr, rmsea, p, r))
            # if rmsea <= self.config.rmsea_tr:
            #     break
        else:
            tr, rmsea, p, r = min(res, key=lambda x: x[1])
        quantile = np.sum(agg_cutcounts < tr) / agg_cutcounts.shape[0]
        print(r, p)
        return FitResults(p.squeeze(), r.squeeze(), rmsea.squeeze(), quantile, tr)

    def fit_for_tr(self, agg_cutcounts, tr):
        agg_cutcounts = agg_cutcounts[agg_cutcounts < tr]
        mean, var = self.estimate_global_mean_and_var(agg_cutcounts, tr=tr)

        p = self.p_from_mean_and_var(mean, var)
        r = self.r_from_mean_and_var(mean, var)
        return p, r
    

    def hist_data_for_tr(self, agg_cutcounts, tr=None):
        if tr is not None:
            agg_cutcounts = agg_cutcounts[agg_cutcounts < tr]
        unique, counts = np.unique(agg_cutcounts, return_counts=True)
        return unique, counts


    def estimate_global_mean_and_var(self, agg_cutcounts: np.ndarray, tr: float):
        nonzero_count = np.count_nonzero(agg_cutcounts)
        has_enough_background = (agg_cutcounts.size > 0) and (nonzero_count / agg_cutcounts.size > self.config.nonzero_windows_to_fit)
        if not has_enough_background:
            self.logger.warning(f"{self.name}: Not enough background to fit the global mean. {nonzero_count}/{agg_cutcounts.shape}. Threshold {tr}. Skipping.")
            raise NoContigPresentError
        
        mean, variance = self.get_mean_and_var(agg_cutcounts)
        return mean, variance

    def calc_rmsea_for_tr(self, obs, unique_cutcounts, p, r, tr, stat='G_sq'):
        assert stat in ('G_sq', 'chi_sq'), "Only G_sq and chi_sq statistics are supported"
        if p <= 0 or p >= 1 or r <= 0:
            return np.inf
        mask = unique_cutcounts < tr
        unique_cutcounts = unique_cutcounts[mask]
        obs = obs[mask].astype(np.float32)
        N = sum(obs)
        exp = st.nbinom.pmf(unique_cutcounts, r, 1 - p) / st.nbinom.cdf(tr - 1, r, 1 - p) * N
        if stat == 'G_sq':
            G_sq = np.sum(calc_g_sq(obs, exp))
        else:
            G_sq = np.sum(calc_chisq(obs, exp))

        df = len(obs) - 2 - 1
        return np.sqrt(np.maximum(G_sq / df - 1, 0) / (N - 1))


class WindowBackgroundFit(BackgroundFit):
    """
    Class to fit the background distribution in a running window fashion
    """
    def fit(self, array: ma.MaskedArray, per_window_trs, global_fit: FitResults=None) -> FitResults:
        agg_cutcounts = array.copy()

        high_signal_mask = (agg_cutcounts >= per_window_trs).filled(False)
        agg_cutcounts[high_signal_mask] = np.nan

        global_r = global_fit.r if global_fit is not None else None
        
        p, r, enough_bg_mask = self.sliding_method_of_moments_fit(
            agg_cutcounts,
            global_r=global_r
        )

        rmsea = np.full_like(p, np.nan, dtype=np.float16)
        per_window_quantiles = np.full_like(p, np.nan, dtype=np.float16)
        return FitResults(
            p, r, rmsea,
            fit_quantile=per_window_quantiles,
            fit_threshold=per_window_trs,
            enough_bg_mask=enough_bg_mask,
        )
    
    def sliding_method_of_moments_fit(self, agg_cutcounts: ma.MaskedArray, global_r=None):
        mean, var = self.sliding_mean_and_variance(agg_cutcounts)
        enough_bg_mask = ~mean.mask

        if global_r is not None:
            r = np.full_like(mean, global_r, dtype=np.float32)
            p = (mean / (mean + r)).filled(np.nan)
        else:
            p = self.p_from_mean_and_var(mean, var).filled(np.nan)
            r = self.r_from_mean_and_var(mean, var).filled(np.nan)
        
        return p, r, enough_bg_mask

    def sliding_mean_and_variance(self, array: ma.MaskedArray, min_count=None, window=None):
        if window is None:
            window = self.config.bg_window

        if min_count is None:
            min_count = self.min_mappable_bg

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
    def running_nanmedian(self, array, window, min_count):
        return bn.move_median(array, window, min_count=min_count).astype(np.float32)
    
    @wrap_masked
    def find_heterogeneous_windows(self, array):
        median_left = self.running_nanmedian(
            array,
            window=self.config.bg_window,
            min_count=self.min_mappable_bg
        )
        median_right = self.running_nanmedian(
            array[::-1],
            window=self.config.bg_window,
            min_count=self.min_mappable_bg
        )[::-1]

        score = np.abs(median_right - median_left)
        outlier_score = np.nanquantile(score, self.config.outlier_detection_tr)
        return score > outlier_score
    
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
    def value_counts_per_bin(self, strided_cutcounts, bin_edges):
        left_edges = bin_edges[0]
        right_edges = bin_edges[1:]
        value_counts = np.full_like(right_edges, 0, dtype=np.float32)
        for i in range(right_edges.shape[0]):
            bin_membership = (strided_cutcounts >= left_edges[None, :]) & (strided_cutcounts < right_edges[i][None, :])
            value_counts[i] = np.sum(bin_membership, axis=0)
            left_edges = right_edges[i]
        
        return value_counts

    def fit_for_bin(self, collapsed_agg_cutcounts, where=None, global_r=None):
        min_count = round(self.config.bg_window * self.config.min_mappable_bg_frac / self.sampling_step)
        enough_bg_mask = np.sum(~np.isnan(collapsed_agg_cutcounts, where=where), axis=0, where=where) > min_count

        
        mean = np.full(collapsed_agg_cutcounts.shape[1], np.nan, dtype=np.float32)
        var = np.full(collapsed_agg_cutcounts.shape[1], np.nan, dtype=np.float32)
        mean[enough_bg_mask], var[enough_bg_mask]  = self.get_mean_and_var(
            collapsed_agg_cutcounts[:, enough_bg_mask],
            where=where[:, enough_bg_mask]
        )

        if global_r is not None:
            r = np.full_like(mean, global_r, dtype=np.float32)
        else:
            r = self.r_from_mean_and_var(mean, var)

        p = mean / (mean + r)

        return p, r, enough_bg_mask
    
    def wrap_rmsea_valid_fits(self, p, r, bin_edges, value_counts, enough_bg_mask=None, n_params=2):
        result = np.full_like(p, np.nan)
        if enough_bg_mask is None:
            enough_bg_mask = np.ones_like(p, dtype=bool)

        result[enough_bg_mask] = self.calc_rmsea_all_windows(
            p[enough_bg_mask],
            r[enough_bg_mask],
            n_params=n_params,
            bin_edges=bin_edges[:, enough_bg_mask],
            value_counts_per_bin=value_counts[:, enough_bg_mask]
        )
        return result
        
    @wrap_masked
    def fit_tr(self, array: ma.MaskedArray, global_fit: FitResults=None):
        original_shape = array.shape
        agg_cutcounts = array[::self.sampling_step]
        best_tr, best_quantile, best_rmsea = self.find_per_window_tr(agg_cutcounts, global_fit=global_fit)
        subsampled_indices = np.arange(
            0, original_shape[0], self.sampling_step, dtype=np.uint32
        )[::self.interpolation_step]
 
        best_tr_with_nan = np.full_like(array, np.nan, dtype=np.float16)
        best_tr_with_nan[subsampled_indices] = best_tr

        best_rmsea_with_nan = np.full_like(array, np.nan, dtype=np.float16)
        best_rmsea_with_nan[subsampled_indices] = best_rmsea

        best_quantile_with_nan = np.full_like(array, np.nan, dtype=np.float16)
        best_quantile_with_nan[subsampled_indices] = best_quantile

        return best_tr_with_nan, best_quantile_with_nan, best_rmsea_with_nan


    def find_per_window_tr(self, agg_cutcounts: np.ndarray, global_fit: FitResults=None):
        strided_agg_cutcounts = rolling_view_with_nan_padding(
            agg_cutcounts,
            points_in_window=self.points_in_bg_window,
            interpolation_step=self.interpolation_step
        ) # shape (bg_window, n_points)
        bin_edges, n_signal_bins = self.get_all_bins(strided_agg_cutcounts)
        value_counts = self.value_counts_per_bin(strided_agg_cutcounts, bin_edges)

        global_r = global_fit.r if global_fit is not None else None

        best_tr = np.asarray(bin_edges[-1], dtype=np.float32)
        remaing_fits_mask = np.ones_like(best_tr, dtype=bool)
        best_rmsea = np.full_like(best_tr, np.inf, dtype=np.float32)
        for i in np.arange(0, n_signal_bins, 1)[::-1]:
            if remaing_fits_mask.sum() == 0:
                break
            current_index = value_counts.shape[0] - i
            right_bin_index = current_index + 1
            mask = strided_agg_cutcounts < bin_edges[right_bin_index - 1, :]

            fit_will_change = value_counts[current_index - 1][remaing_fits_mask] != 0 # shape of remaining_fits
            
            changing_indices = np.where(remaing_fits_mask)[0][fit_will_change]

            p, r, enough_bg_mask = self.fit_for_bin(
                strided_agg_cutcounts[:, changing_indices],
                where=mask[:, changing_indices],
                global_r=global_r
            )
            
            n_params = 1 if global_r is not None else 2
            

            edges = bin_edges[:right_bin_index, changing_indices]
            counts = value_counts[:current_index, changing_indices]

            rmsea = self.wrap_rmsea_valid_fits(
                p, r,
                edges, counts, 
                enough_bg_mask,
                n_params=n_params
            ) # shape of r
            
            successful_fits = ~enough_bg_mask | (rmsea <= self.config.rmsea_tr)
            # best fit found
            better_fit = np.where(
                (rmsea < best_rmsea[changing_indices]),
                True,
                False
            )

            best_tr[changing_indices] = np.where(
                better_fit,
                edges[-1],
                best_tr[changing_indices]
            )
            best_rmsea[changing_indices] = np.where(
                better_fit,
                rmsea,
                best_rmsea[changing_indices]
            )

            # Not enough background
            best_tr[changing_indices] = np.where(
                ~enough_bg_mask,
                np.nan,
                best_tr[changing_indices]
            )
            best_rmsea[changing_indices] = np.where(
                ~enough_bg_mask,
                np.nan,
                best_rmsea[changing_indices]
            )

            remaing_fits_mask[changing_indices] = ~successful_fits

            idx = n_signal_bins - i
            if idx % (n_signal_bins // 5) == 0:
                self.logger.debug(f"{self.name} (window={self.config.bg_window}): Identifying signal proportion (step {idx}/{n_signal_bins})")
        
        if global_fit is not None:
            global_quantile_tr = self.get_bg_tr(strided_agg_cutcounts, global_fit.fit_quantile)
            best_tr = np.where(
                best_rmsea <= 0.1,
                best_tr,
                np.maximum(global_quantile_tr, global_fit.fit_threshold),
            )
        with np.errstate(invalid='ignore'):
            best_quantile = np.sum(strided_agg_cutcounts < best_tr, axis=0) / np.sum(~np.isnan(strided_agg_cutcounts), axis=0)
        return best_tr, best_quantile, best_rmsea

    @wrap_masked
    def calc_rmsea_all_windows(
        self,
        p: np.ndarray,
        r: np.ndarray,
        n_params: int,
        bin_edges, # edges of the bins, right edge not inclusive
        value_counts_per_bin, # number of observed cutcounts in each bin for each window
    ):
        """
        Calculate RMSEA for all sliding windows.
        """
        bg_sum_mappable = np.sum(value_counts_per_bin, axis=0)
        # print(bin_edges)
        #sf_values = st.nbinom.sf(bin_edges - 1, r, 1 - p)
        sf_values = np.where(bin_edges == 0, 1., betainc(bin_edges, r, p))
        sf_diffs = -np.diff(sf_values, axis=0)
        assert sf_diffs.shape == value_counts_per_bin.shape, f"SF diffs shape should match value counts shape. Got SF: {sf_diffs.shape} and vc: {value_counts_per_bin.shape}"
        norm_coef = 1 - sf_values[-1]
        expected_counts = (sf_diffs * bg_sum_mappable / norm_coef)
        G_sq = np.sum(
            calc_g_sq(value_counts_per_bin, expected_counts), 
            axis=0
        )
        df = np.sum(
            (value_counts_per_bin > 0) & (np.diff(bin_edges, axis=0) != 0),
            axis=0
        ) - n_params - 1

        G_sq = np.divide(G_sq, df, out=np.zeros_like(G_sq), where=df >= 7)

        rmsea = np.sqrt(np.maximum(G_sq - 1, 0) / (bg_sum_mappable - 1))
        rmsea = np.where(df >= 7, rmsea, np.inf)
        assert np.sum(np.isnan(rmsea)) == 0, "RMSEA should not contain NaNs"
        return rmsea
