import numpy.ma as ma
import numpy as np
from scipy import stats as st
from scipy.special import betainc
from hotspot3.models import NoContigPresentError, ProcessorConfig, GlobalFitResults, WindowedFitResults
from hotspot3.connectors.bottleneck import BottleneckWrapper
from hotspot3.utils import wrap_masked, rolling_view_with_nan_padding
from hotspot3.stats import calc_g_sq, calc_chisq, calc_rmsea


class BackgroundFit(BottleneckWrapper):
    def __init__(self, config: ProcessorConfig=None, logger=None, name=None):
        super().__init__(logger=logger, config=config, name=name)

        self.min_mappable_bg = round(self.config.min_mappable_bg_frac * self.config.bg_window)
        self.sampling_step = self.config.signal_prop_sampling_step
        self.interpolation_step = self.config.signal_prop_interpolation_step // self.sampling_step
        self.points_in_bg_window = (self.config.bg_window - 1) // self.sampling_step
        if self.points_in_bg_window % 2 == 0:
            self.points_in_bg_window += 1

    def fit(self):
        raise NotImplementedError
    
    @wrap_masked
    def get_mean_and_var(self, array: np.ndarray, **kwargs):
        mean = np.nanmean(array, axis=0, dtype=np.float32, **kwargs)
        var = np.nanvar(array, ddof=1, axis=0, dtype=np.float32, **kwargs)
        return mean, var

    @wrap_masked
    def p_from_mean_and_r(self, mean: np.ndarray, r: np.ndarray, where=None):
        with np.errstate(divide='ignore', invalid='ignore', all='ignore'):
            p = np.array(mean / (mean + r), dtype=np.float32)
        return p
    
    @wrap_masked
    def r_from_mean_and_var(self, mean: np.ndarray, var: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore', all='ignore'):
            r = np.array(mean ** 2 / (var - mean), dtype=np.float32)
        return r
    
    def get_signal_bins(self, array: np.ndarray, min_bg_tr=None):
        if min_bg_tr is None:
            min_bg_tr = self.quantile_ignore_all_na(array, self.config.min_background_prop)
        max_bg_tr = self.quantile_ignore_all_na(array, self.config.max_background_prop)
        n_signal_bins = min(np.nanmax(max_bg_tr - min_bg_tr), self.config.num_signal_bins)
        n_signal_bins = round(n_signal_bins)
        
        result = np.full((n_signal_bins + 1, *min_bg_tr.shape), np.nan, dtype=np.float32)
        nan_tr = np.isnan(min_bg_tr) | np.isnan(max_bg_tr)
        result[:, ~nan_tr] = np.round(
            np.linspace(min_bg_tr[~nan_tr], max_bg_tr[~nan_tr], n_signal_bins + 1)
        )
        return result, n_signal_bins

    def quantile_ignore_all_na(self, array: np.ndarray, quantile: float):
        """
        Wrapper of nan quantile to avoid runtime warnings when data is all nan.
        """
        all_nan = np.all(np.isnan(array), axis=0)
        if array.ndim == 1:
            return np.nan if all_nan else np.nanquantile(array, quantile)
        result = np.full(array.shape[1], np.nan, dtype=np.float32)
        result[~all_nan] = np.nanquantile(array[:, ~all_nan], quantile, axis=0)
        return result


class GlobalBackgroundFit(BackgroundFit):
    """
    Class to fit the background distribution globally (for chromosome/homogeneous regions)
    """
    def fit(self, agg_cutcounts: ma.MaskedArray, step=None, global_fit: GlobalFitResults = None) -> GlobalFitResults:
        """
        Fit the global background distribution.

        Parameters:
            agg_cutcounts (np.ndarray): Array of aggregated cutcounts.
            step (int): Step to reduce computational burden and improve speed. Can be set to 1 for full resolution.
        """
        result = []
        agg_cutcounts = agg_cutcounts.filled(np.nan)
        trs, _ = self.get_signal_bins(agg_cutcounts)
        for tr in trs:
            try:
                p, r, rmsea = self.fit_for_tr(agg_cutcounts, tr, step=step, global_fit=global_fit)
            except NoContigPresentError:
                continue
            result.append((tr, rmsea, p, r))
        
        tr, rmsea, p, r = min(result, key=lambda x: x[1])
        if rmsea > self.config.rmsea_tr:
            self.logger.warning(f"{self.name}: RMSEA ({rmsea}) is too high for the best fit. Using {(self.config.max_background_prop * 100):.2f}% as threshold.")
            tr, rmsea, p, r = result[-1]
        quantile = np.sum(agg_cutcounts < tr) / np.sum(~np.isnan(agg_cutcounts))

        return GlobalFitResults(p, r, rmsea, quantile, tr)

    def fit_for_tr(self, agg_cutcounts, tr, step=None, global_fit: GlobalFitResults=None):
        if step is None:
            step = 1
        assumed_signal_mask = self.filter_by_tr_spatially(agg_cutcounts, tr)
        filtered_agg_cutcounts = agg_cutcounts[~assumed_signal_mask & ~np.isnan(agg_cutcounts)][::step]

        mean, var = self.estimate_global_mean_and_var(filtered_agg_cutcounts)
        if global_fit is not None:
            r = global_fit.r
        else:
            r = self.r_from_mean_and_var(mean, var)
        p = self.p_from_mean_and_r(mean, var)


        unique, counts = np.unique(filtered_agg_cutcounts, return_counts=True)
        rmsea = self.calc_rmsea_for_tr(counts, unique, p, r, tr)
    
        return p, r, rmsea

    def estimate_global_mean_and_var(self, agg_cutcounts: np.ndarray):
        nonzero_count = np.count_nonzero(agg_cutcounts)
        has_enough_background = (agg_cutcounts.size > 0) and (nonzero_count / agg_cutcounts.size > self.config.nonzero_windows_to_fit)
        if not has_enough_background:
            self.logger.warning(f"{self.name}: Not enough background to fit the global mean. {nonzero_count}/{agg_cutcounts.shape}")
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
        return calc_rmsea(G_sq, N, df)


class WindowBackgroundFit(BackgroundFit):
    """
    Class to fit the background distribution in a running window fashion
    """
    def fit(self, array: ma.MaskedArray, per_window_trs, global_fit: GlobalFitResults=None) -> WindowedFitResults:
        agg_cutcounts = array.copy()

        high_signal_mask = self.filter_by_tr_spatially(agg_cutcounts, per_window_trs)
        agg_cutcounts[high_signal_mask] = np.nan

        global_r = global_fit.r if global_fit is not None else None
        
        p, r, enough_bg_mask = self.sliding_method_of_moments_fit(
            agg_cutcounts,
            global_r=global_r
        )

        return WindowedFitResults(p, r, enough_bg_mask=enough_bg_mask)
    
    def sliding_method_of_moments_fit(self, agg_cutcounts: ma.MaskedArray, global_r=None):
        mean, var = self.sliding_mean_and_variance(agg_cutcounts)
        enough_bg_mask = ~mean.mask

        if global_r is not None:
            r = np.full_like(mean, global_r, dtype=np.float32)
        else:
            r = self.r_from_mean_and_var(mean, var).filled(np.nan)

        p = self.p_from_mean_and_r(mean, r).filled(np.nan)
        return p, r, enough_bg_mask

    def sliding_mean_and_variance(self, array: ma.MaskedArray, window=None):
        if window is None:
            window = self.config.bg_window

        mean = self.centered_running_nanmean(array, window)
        var = self.centered_running_nanvar(array, window)
        mean = ma.masked_invalid(mean)
        var = ma.masked_invalid(var)
        return mean, var
    
    @wrap_masked
    def find_heterogeneous_windows(self, array):
        median_left = self.running_nanmedian(
            array,
            window=self.config.bg_window,
        )
        median_right = self.running_nanmedian(
            array[::-1],
            window=self.config.bg_window,
        )[::-1]

        score = np.abs(median_right - median_left)
        outlier_score = np.nanquantile(score, self.config.outlier_detection_tr)
        return score > outlier_score


class StridedBackgroundFit(BackgroundFit):
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
        window = min(self.config.bg_window, collapsed_agg_cutcounts.shape[0])
        min_count = self.get_min_count(window / self.sampling_step)
        enough_bg_mask = np.sum(
            ~np.isnan(collapsed_agg_cutcounts, where=where),
            axis=0,
            where=where
        ) > min_count

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

        p = self.p_from_mean_and_r(mean, r)

        return p, r, enough_bg_mask

    def get_all_bins(self, array: np.ndarray):
        min_bg_tr = self.quantile_ignore_all_na(array, self.config.min_background_prop)
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
    def fit_tr(self, array: ma.MaskedArray, global_fit: GlobalFitResults=None):
        original_shape = array.shape
        strided_agg_cutcounts = rolling_view_with_nan_padding(
            array[::self.sampling_step],
            points_in_window=self.points_in_bg_window,
            interpolation_step=self.interpolation_step
        ) # shape (bg_window, n_points)
        best_tr, best_quantile, best_rmsea = self.find_per_window_tr(
            strided_agg_cutcounts,
            global_fit=global_fit,
        )
    
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


    def find_per_window_tr(self, strided_agg_cutcounts: np.ndarray, global_fit: GlobalFitResults=None):
        bin_edges, n_signal_bins = self.get_all_bins(strided_agg_cutcounts)
        value_counts = self.value_counts_per_bin(strided_agg_cutcounts, bin_edges)

        step = round(self.config.exclude_peak_flank_length / self.sampling_step)
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
            for x in range(1, (step - 1) // 2):
                mask[:-x, :] &= mask[x:, :]
                mask[x:, :] &= mask[:-x, :] 

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
            best_tr = np.where(
                best_rmsea <= self.config.rmsea_tr,
                best_tr, 
                self.fallback_tr(strided_agg_cutcounts, global_fit, best_tr)
            )
        with np.errstate(invalid='ignore'):
            best_quantile = np.sum(strided_agg_cutcounts < best_tr, axis=0) / np.sum(~np.isnan(strided_agg_cutcounts), axis=0)
        return best_tr, best_quantile, best_rmsea

    def fallback_tr(self, strided_agg_cutcounts, global_fit: GlobalFitResults, best_tr: np.ndarray):
        global_quantile_tr = self.quantile_ignore_all_na(strided_agg_cutcounts, global_fit.fit_quantile)
        return np.where(
            global_quantile_tr < global_fit.fit_threshold,
            global_fit.fit_threshold,
            np.where(
                best_tr < global_quantile_tr,
                best_tr,
                self.quantile_ignore_all_na(strided_agg_cutcounts, self.config.min_background_prop) # FIXME, already calc in bin_edges
            )
        )


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
        return calc_rmsea(G_sq, bg_sum_mappable, df)
