"""
Low level classes to fit background.
GlobalBackgroundFit: Fit the whole array.
WindowBackgroundFit: Fit the array in a windowed fashion.
StridedBackgroundFit: Fit the array using a strided approach (takes a lot of memory though).
"""

import numpy.ma as ma
import numpy as np
from scipy.special import betainc
from typing import List
import dataclasses

from hotspot3.models import NotEnoughDataForContig, FitResults, WindowedFitResults, DataForFit
from hotspot3.config import ProcessorConfig
from hotspot3.connectors.bottleneck import BottleneckWrapper
from hotspot3.background_fit import calc_rmsea, check_valid_fit, rolling_view_with_nan_padding
from hotspot3.utils import wrap_masked


class BackgroundFit(BottleneckWrapper):
    def __init__(self, config: ProcessorConfig=None, logger=None, name=None):
        super().__init__(logger=logger, config=config, name=name)

        self.sampling_step = self.config.signal_prop_sampling_step
        self.interpolation_step = self.config.signal_prop_interpolation_step // self.sampling_step

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
            p = np.array(mean / (mean + r), dtype=np.float32) # check if we return p or 1-p
        return p
    
    @wrap_masked
    def r_from_mean_and_var(self, mean: np.ndarray, var: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore', all='ignore'):
            r = np.array(mean ** 2 / (var - mean), dtype=np.float32)
        return r
    
    def value_counts_per_bin(self, array, bin_edges):
        left_edges = bin_edges[0]
        right_edges = bin_edges[1:]
        value_counts = np.full_like(right_edges, 0, dtype=np.float32)
        for i in range(right_edges.shape[0]):
            bin_membership = (array >= left_edges[None, :]) & (array < right_edges[i][None, :])
            value_counts[i] = np.sum(bin_membership, axis=0)
            left_edges = right_edges[i]
        
        return value_counts
    
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

    def get_signal_bins(self, array: np.ndarray, min_bg_tr=None):
        if min_bg_tr is None:
            min_bg_tr = self.quantile_ignore_all_na(array, self.config.min_background_prop)
        max_bg_tr = self.quantile_ignore_all_na(array, self.config.max_background_prop)
        n_signal_bins = min(np.nanmax(max_bg_tr - min_bg_tr), self.config.num_signal_bins)
        n_signal_bins = round(n_signal_bins)
        
        bin_edges = np.full((n_signal_bins + 1, *min_bg_tr.shape), np.nan, dtype=np.float32)
        nan_tr = np.isnan(min_bg_tr) | np.isnan(max_bg_tr)
        bin_edges[:, ~nan_tr] = np.round(
            np.linspace(min_bg_tr[~nan_tr], max_bg_tr[~nan_tr], n_signal_bins + 1)
        )

        return bin_edges, n_signal_bins
    
    def merge_signal_bins_for_rmsea_inplace(self, value_counts, bin_edges, n_signal_bins):
        assert bin_edges.shape[0] == value_counts.shape[0] + 1, f"Bin edges shape should be one more than value counts shape. Got {bin_edges.shape} and {value_counts.shape}"
        for j in np.arange(n_signal_bins):
            i = value_counts.shape[0] - j - 1
            if value_counts[i] < self.config.min_obs_rmsea:
                value_counts[i - 1] += value_counts[i]
                value_counts[i] = 0
                bin_edges[i] = bin_edges[i + 1]

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
    
    @wrap_masked
    def calc_rmsea_all_windows(
        self,
        p: np.ndarray,
        r: np.ndarray,
        n_params: int,
        bin_edges: np.ndarray, # edges of the bins, right edge not inclusive
        value_counts_per_bin: np.ndarray, # number of observed cutcounts in each bin for each window
    ):
        """
        Calculate RMSEA for all sliding windows.
        """
        valid_bins = value_counts_per_bin >= self.config.min_obs_rmsea
        bg_sum_mappable = np.sum(value_counts_per_bin, axis=0, where=valid_bins)
        # print(bin_edges)
        #sf_values = st.nbinom.sf(bin_edges - 1, r, 1 - p)
        sf_values = np.where(bin_edges == 0, 1., betainc(bin_edges, r, p))
        sf_diffs = -np.diff(sf_values, axis=0)
        assert sf_diffs.shape == value_counts_per_bin.shape, f"SF diffs shape should match value counts shape. Got SF: {sf_diffs.shape} and vc: {value_counts_per_bin.shape}"
        norm_coef = 1 - sf_values[-1]
        expected_counts = (sf_diffs * bg_sum_mappable / norm_coef)
        df = np.sum(
            np.diff(bin_edges, axis=0) != 0,
            axis=0,
            where=valid_bins
        ) - n_params - 1
        return calc_rmsea(value_counts_per_bin, expected_counts, bg_sum_mappable, df, where=valid_bins)

    def get_bg_quantile_from_tr(self, agg_cutcounts: np.ndarray, tr: float):
        with np.errstate(invalid='ignore'):
            result = np.sum(agg_cutcounts < tr, axis=0) / np.sum(~np.isnan(agg_cutcounts), axis=0)
        return result


class GlobalBackgroundFit(BackgroundFit):
    """
    Class to fit the background distribution globally (for chromosome/homogeneous regions)
    """
    def fit(
            self,
            agg_cutcounts: ma.MaskedArray,
            step=None,
            fallback_fit_results: FitResults = None
        ) -> FitResults:
        """
        Fit the global background distribution.

        Parameters:
            agg_cutcounts (np.ndarray): Array of aggregated cutcounts.
            step (int): Step to reduce computational burden and improve speed. Can be set to 1 for full resolution.
        """
        
        data_for_fit = self.prepare_data_for_fit(agg_cutcounts, step)
        result = self.fit_all_thresholds(data_for_fit, None)
        if len(result) == 0:
            raise NotEnoughDataForContig
        
        best_fit_result = min(result, key=lambda x: x.rmsea)

        best_fit_result = self.fallback_suspicious_fit(
            data_for_fit,
            best_fit_result,
            fallback_fit_results
        )

        return best_fit_result
    
    def fallback_suspicious_fit(
            self,
            data_for_fit: DataForFit,
            best_fit_result: FitResults,
            fallback_fit_results: FitResults
        ):
        #FIXME: it is bullshit
        if best_fit_result.rmsea > self.config.rmsea_tr and fallback_fit_results is not None:
            chrom_quantile_tr = np.nanquantile(data_for_fit.agg_cutcounts, fallback_fit_results.fit_quantile)
            if chrom_quantile_tr < fallback_fit_results.fit_threshold:
                tr = fallback_fit_results.fit_threshold
                best_fit_result = self.fit_for_tr(
                    data_for_fit,
                    tr=tr,
                    fallback_fit_results=fallback_fit_results
                )
            else:
                pass

        if not check_valid_fit(best_fit_result) and fallback_fit_results is not None:
            tr = fallback_fit_results.fit_threshold # FIXME for loop to find the best tr
            best_fit_result = self.fit_for_tr(
                data_for_fit,
                tr=tr,
                fallback_fit_results=fallback_fit_results
            )
        return best_fit_result
    
    def prepare_data_for_fit(self, agg_cutcounts: ma.MaskedArray, step: int):
        agg_cutcounts = agg_cutcounts.filled(np.nan)
        max_counts_with_flanks = self.get_max_count_with_flanks(agg_cutcounts)[::step]
        agg_cutcounts = agg_cutcounts[::step]
        bin_edges, n_signal_bins = self.get_all_bins(agg_cutcounts)
        bin_edges = bin_edges[:, None]
        value_counts = self.value_counts_per_bin(agg_cutcounts[:, None], bin_edges)
        self.merge_signal_bins_for_rmsea_inplace(value_counts, bin_edges, n_signal_bins)
        return DataForFit(bin_edges, value_counts, n_signal_bins, agg_cutcounts, max_counts_with_flanks)
    
    def fit_all_thresholds(
            self,
            data_for_fit: DataForFit,
            fallback_fit_results: FitResults=None
            ) -> List[FitResults]:
        result = []
        for i in np.arange(data_for_fit.n_signal_bins)[::-1]:
            tr = data_for_fit.bin_edges[len(data_for_fit.bin_edges) - i - 1]
            #self.logger.debug(f"{self.name}: Attempting global fit at tr={tr}")
            try:
                assumed_signal_mask = data_for_fit.max_counts_with_flanks >= tr
                p, r, rmsea = self._fit_for_bg_mask(
                    data_for_fit.agg_cutcounts,
                    bin_edges=data_for_fit.bin_edges,
                    assumed_signal_mask=assumed_signal_mask,
                    fallback_fit_results=fallback_fit_results
                )
            except NotEnoughDataForContig:
                continue
            #FIXME: remove if slow
            q = self.get_bg_quantile_from_tr(data_for_fit.agg_cutcounts, tr)
            result.append(FitResults(p, r, rmsea, q, tr))
        return result
    
    @wrap_masked
    def fit_for_tr(
            self,
            data_for_fit: DataForFit,
            tr: float,
            fallback_fit_results: FitResults=None
    ):
        assumed_signal_mask = data_for_fit.max_counts_with_flanks >= tr

        p, r, rmsea = self._fit_for_bg_mask(
            data_for_fit.agg_cutcounts,
            assumed_signal_mask,
            data_for_fit.bin_edges,
            fallback_fit_results=fallback_fit_results,
        )
        q = self.get_bg_quantile_from_tr(data_for_fit.agg_cutcounts, tr)
        return FitResults(p, r, rmsea, q, tr)
    
    def _fit_for_bg_mask(
        self,
        agg_cutcounts,
        assumed_signal_mask,
        bin_edges,
        fallback_fit_results: FitResults=None
    ):
        mean, var = self.get_mean_and_var(agg_cutcounts, where=~assumed_signal_mask)
        if fallback_fit_results is not None:
            r = fallback_fit_results.r
            n_params = 1
        else:
            r = self.r_from_mean_and_var(mean, var)
            n_params = 2
        p = self.p_from_mean_and_r(mean, r)

        if not check_valid_fit(FitResults(p, r, 0, 0, 0)):
            raise NotEnoughDataForContig
        
        value_counts = self.value_counts_per_bin(agg_cutcounts[~assumed_signal_mask, None], bin_edges)
        rmsea = self.calc_rmsea_all_windows(
            p,
            r,
            n_params,
            bin_edges,
            value_counts,
        )[0]
        return p, r, rmsea


class StridedBackgroundFit(BackgroundFit):
    """
    Class to fit the background distribution using a strided approach.
    Extemely computationally taxing, use large stride values to compensate.
    """
    @wrap_masked
    def fit(self, agg_cutcounts: np.ndarray, fallback_fit_results: FitResults=None):
        data_for_fit = self.prepare_data_for_fit(agg_cutcounts)

        best_fit_results = self.find_best_fits(
            data_for_fit,
            fallback_fit_results=fallback_fit_results,
        )

        if fallback_fit_results is not None:
            best_fit_results.fit_threshold = np.where(
                best_fit_results.rmsea <= self.config.rmsea_tr,
                best_fit_results.fit_threshold , 
                self.fallback_tr(data_for_fit, best_fit_results, fallback_fit_results)
            )

        return self.cast_to_original_shape(best_fit_results, agg_cutcounts.shape)
    
    def get_strided_agg_cutcounts(self, agg_cutcounts: np.ndarray):
        points_in_bg_window = (self.config.bg_window - 1) // self.sampling_step
        if points_in_bg_window % 2 == 0:
            points_in_bg_window += 1
        
        strided_agg_cutcounts = rolling_view_with_nan_padding(
            agg_cutcounts[::self.sampling_step],
            points_in_window=points_in_bg_window,
            interpolation_step=self.interpolation_step
        ) # shape (bg_window, n_points)

        return strided_agg_cutcounts
    
    def cast_to_original_shape(
            self,
            fit_results: FitResults,
            original_shape: tuple
        ) -> FitResults:
        subsampled_indices = np.arange(
            0, original_shape[0], self.sampling_step, dtype=np.uint32
        )[::self.interpolation_step]

        for field in dataclasses.fields(fit_results):
            values = getattr(fit_results, field)
            if values is not None and not (np.ndim(values) == 0 and np.isnan(values)):
                upcasted = np.full(original_shape, np.nan, dtype=np.float16)
                upcasted[subsampled_indices] = values
                setattr(fit_results, field, upcasted)
        return fit_results
    
    def prepare_data_for_fit(self, agg_cutcounts: ma.MaskedArray):
        strided_agg_cutcounts = self.get_strided_agg_cutcounts(agg_cutcounts)
        max_counts_with_flanks = self.get_max_count_with_flanks(agg_cutcounts)
        strided_max_counts_with_flanks = self.get_strided_agg_cutcounts(max_counts_with_flanks)
        bin_edges, n_signal_bins = self.get_all_bins(strided_agg_cutcounts)
        value_counts = self.value_counts_per_bin(strided_agg_cutcounts, bin_edges)
        self.merge_signal_bins_for_rmsea_inplace(value_counts, bin_edges, n_signal_bins)
        return DataForFit(bin_edges, value_counts, n_signal_bins, strided_agg_cutcounts, strided_max_counts_with_flanks)


    def find_best_fits(
            self,
            data_for_fit: DataForFit,
            fallback_fit_results: FitResults=None
        ) -> FitResults:

        best_fit_results = FitResults(
            None,
            None,
            rmsea=np.full(data_for_fit.agg_cutcounts.shape[1], np.inf, dtype=np.float32),
            fit_threshold=np.asarray(data_for_fit.bin_edges[-1], dtype=np.float32),
        )
        remaing_fits_mask = np.ones_like(best_fit_results.fit_threshold, dtype=bool)

        for i in np.arange(0, data_for_fit.n_signal_bins, 1)[::-1]:
            if remaing_fits_mask.sum() == 0:
                break

            # 1. Prepare data for the current threshold
            current_index = data_for_fit.bin_edges.shape[0] - i - 1
            changing_indices = self._get_changing_indices(current_index, data_for_fit, remaing_fits_mask)

            data_for_current_threshold = self.get_data_for_current_threshold(
                data_for_fit,
                current_index,
                changing_indices
            )

            current_thresholds = data_for_current_threshold.bin_edges[-1]
            assumed_signal_mask = data_for_current_threshold.max_counts_with_flanks >= current_thresholds
            
            # 2. Fit for the current threshold and evaluate the fit
            bin_fit_results = self.fit_for_bin(
                data_for_current_threshold.agg_cutcounts,
                where=~assumed_signal_mask,
                fallback_fit_results=fallback_fit_results
            )
            enough_bg_mask = bin_fit_results.enough_bg_mask
            bin_fit_results = FitResults(
                bin_fit_results.p,
                bin_fit_results.r,
            )

            bin_fit_results.rmsea = self.evaluate_fit_for_bin(
                data_for_current_threshold,
                bin_fit_results,
                fallback_fit_results,
            )
            bin_fit_results.fit_threshold = current_thresholds

            # 3. Update the best fit results
            self.update_remaining_fits(
                best_fit_results,
                remaing_fits_mask,
                bin_fit_results,
                enough_bg_mask,
                changing_indices
            )

            idx = data_for_fit.n_signal_bins - i
            if idx % (max(data_for_fit.n_signal_bins, 5) // 5) == 0:
                self.logger.debug(f"{self.name} (window={self.config.bg_window}): Identifying signal proportion (step {idx}/{data_for_fit.n_signal_bins})")
        
        best_fit_results.fit_quantile = self.get_bg_quantile_from_tr(data_for_fit.agg_cutcounts, best_fit_results.fit_threshold)
        return best_fit_results
    
    def get_data_for_current_threshold(
            self,
            data_for_fit: DataForFit,
            current_index,
            changing_indices
    ):
        right_bin_index = current_index + 1

        current_agg_cutcounts = data_for_fit.agg_cutcounts[:, changing_indices]
        current_bin_edges = data_for_fit.bin_edges[:right_bin_index, changing_indices]

        # TODO:
        # current_value_counts = self.value_counts_per_bin(
        #     current_agg_cutcounts,
        #     current_bin_edges,
        #     where=~assumed_signal_mask
        # )

        current_value_counts = data_for_fit.value_counts[:current_index, changing_indices]

        return DataForFit(
            current_bin_edges,
            current_value_counts,
            data_for_fit.n_signal_bins,
            current_agg_cutcounts,
            data_for_fit.max_counts_with_flanks[:, changing_indices]
        )
        
    
    def _get_changing_indices(self, current_index, data_for_fit: DataForFit, remaing_fits_mask: np.ndarray):
        fit_will_change = data_for_fit.value_counts[current_index - 1][remaing_fits_mask] != 0 # shape of remaining_fits
        changing_indices = np.where(remaing_fits_mask)[0][fit_will_change]
        return changing_indices
    
    def fit_for_bin(
            self,
            strided_agg_cutcounts: np.ndarray,
            where: np.ndarray=None,
            fallback_fit_results: FitResults=None
        ):
        window = min(self.config.bg_window, strided_agg_cutcounts.shape[0])
        min_count = self.get_min_count(window / self.sampling_step)
        enough_bg_mask = np.sum(
            ~np.isnan(strided_agg_cutcounts, where=where),
            axis=0,
            where=where
        ) > min_count

        mean = np.full(strided_agg_cutcounts.shape[1], np.nan, dtype=np.float32)
        var = np.full(strided_agg_cutcounts.shape[1], np.nan, dtype=np.float32)
        mean[enough_bg_mask], var[enough_bg_mask] = self.get_mean_and_var(
            strided_agg_cutcounts[:, enough_bg_mask],
            where=where[:, enough_bg_mask]
        )

        if fallback_fit_results is not None:
            r = np.full_like(mean, fallback_fit_results.r, dtype=np.float32)
        else:
            r = self.r_from_mean_and_var(mean, var)

        p = self.p_from_mean_and_r(mean, r)

        return WindowedFitResults(p, r, enough_bg_mask=enough_bg_mask)
    
    def evaluate_fit_for_bin(
            self,
            data_for_fit: DataForFit,
            fit_results: WindowedFitResults,
            fallback_fit_results: FitResults,
        ):
        n_params = 1 if fallback_fit_results is not None else 2
        rmsea = np.full_like(fit_results.p, np.nan)

        rmsea[fit_results.enough_bg_mask] = self.calc_rmsea_all_windows(
            fit_results.p[fit_results.enough_bg_mask],
            fit_results.r[fit_results.enough_bg_mask],
            n_params=n_params,
            bin_edges=data_for_fit.bin_edges[:, fit_results.enough_bg_mask],
            value_counts_per_bin=data_for_fit.value_counts[:, fit_results.enough_bg_mask]
        )
        return rmsea
    
    def update_remaining_fits(
            self,
            best_fit_results: FitResults,
            remaing_fits_mask: np.ndarray,
            bin_fit_results: FitResults,
            enough_bg_mask: np.ndarray,
            changing_indices: np.ndarray,
        ):
        remaining_fits = enough_bg_mask & ~(bin_fit_results.rmsea <= self.config.rmsea_tr)
        remaing_fits_mask[changing_indices] = remaining_fits

        better_fit = bin_fit_results.rmsea <= best_fit_results.rmsea[changing_indices]
        self.update_best_fit(
            best_fit_results, bin_fit_results,
            changing_indices,
            where=better_fit
        )
        self.update_best_fit(
            best_fit_results,
            FitResults(),
            changing_indices,
            where=~enough_bg_mask
        )
    
    def update_best_fit(
            self,
            best_fit, current_fit, changing_indices, where
        ): 
        for field in ['rmsea', 'fit_threshold']:
            current = getattr(current_fit, field)
            best = getattr(best_fit, field)[changing_indices]
            best = np.where(
                where,
                current,
                best
            )

    def fallback_tr(
            self,
            data_for_fit: DataForFit,
            fit_results: FitResults,
            fallback_fit_results: FitResults
        ):
        global_quantile_tr = self.quantile_ignore_all_na(
            data_for_fit.agg_cutcounts,
            fallback_fit_results.fit_quantile
        )
        return np.where(
            global_quantile_tr < fallback_fit_results.fit_threshold,
            fallback_fit_results.fit_threshold,
            np.where(
                fit_results.fit_threshold < global_quantile_tr,
                fit_results.fit_threshold,
                self.quantile_ignore_all_na(data_for_fit.agg_cutcounts, self.config.min_background_prop) # FIXME, already calc in bin_edges
            )
        )
    

class WindowBackgroundFit(BackgroundFit):
    """
    Class to fit the background distribution in a running window fashion
    """
    def fit(self, array: ma.MaskedArray, per_window_trs, global_fit: FitResults=None) -> WindowedFitResults:
        agg_cutcounts = array.copy()

        high_signal_mask = self.get_max_count_with_flanks(agg_cutcounts) >= per_window_trs
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
