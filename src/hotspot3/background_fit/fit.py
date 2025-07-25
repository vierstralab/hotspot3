"""
Low level classes to fit background.
GlobalBackgroundFit: Fit the whole array.
WindowBackgroundFit: Fit the array in a windowed fashion.
StridedBackgroundFit: Fit the array using a strided approach (core functionality is now defunc).
Currently, it is used only to calculate sliding quantile
"""

import numpy.ma as ma
import numpy as np
from scipy.special import betainc
from typing import List

from hotspot3.helpers.models import NotEnoughDataForContig, FitResults, WindowedFitResults, DataForFit
from hotspot3.config import ProcessorConfig
from hotspot3.connectors.bottleneck import BottleneckWrapper
from hotspot3.background_fit import calc_rmsea, rolling_view_with_nan_padding
from hotspot3.helpers.stats import check_valid_nb_params
from hotspot3.helpers.utils import wrap_masked


class BackgroundFit(BottleneckWrapper):
    def __init__(self, config: ProcessorConfig=None, logger=None, name=None):
        super().__init__(logger=logger, config=config, name=name)

        self.sampling_step = self.config.signal_prop_sampling_step
        self.interpolation_step = self.config.signal_prop_interpolation_step // self.sampling_step

    def fit(self):
        raise NotImplementedError
    
    def prepare_data_for_fit(self, array: np.ndarray):
        raise NotImplementedError
    
    @wrap_masked
    def get_mean_and_var(self, array: np.ndarray, **kwargs):
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.nanmean(array, axis=0, dtype=np.float32, **kwargs)
            var = np.nanvar(array, ddof=1, axis=0, dtype=np.float32, **kwargs)
        return mean, var

    @wrap_masked
    def p_from_mean_and_r(self, mean: np.ndarray, r: np.ndarray, where=None):
        with np.errstate(divide='ignore', invalid='ignore', all='ignore'):
            p = np.array(mean / (mean + r), dtype=np.float16) # check if we return p or 1-p
        return p
    
    @wrap_masked
    def r_from_mean_and_var(self, mean: np.ndarray, var: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore', all='ignore'):
            r = np.array(mean ** 2 / (var - mean), dtype=np.float16)
        return r
    
    def value_counts_per_bin(self, array, bin_edges, where=None):
        if where is None:
            where = np.ones(array.shape, dtype=bool)
        left_edge = bin_edges[0]
        right_edges = bin_edges[1:]
        value_counts = np.full_like(right_edges, 0, dtype=np.float32)
        for i in range(right_edges.shape[0]):
            right_edge = right_edges[i]
            bin_membership = (array >= left_edge[None, :]) & (array < right_edge[None, :])
            value_counts[i] = np.sum(bin_membership, axis=0, where=where)
            left_edge = right_edge
        
        return value_counts
    
    def get_all_bins(self, array: np.ndarray, fallback_fit_results: FitResults=None):
        print("No warning1")
        min_bg_tr = self.quantile_ignore_all_na(array, self.config.min_background_prop)
        signal_bins, n_signal_bins = self.get_signal_bins(
            array,
            min_bg_tr=min_bg_tr,
            fallback_fit_results=fallback_fit_results
        )
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
        print("No warning1?")
        
        return np.concatenate([bg_bins[:-1], signal_bins]), n_signal_bins

    def get_signal_bins(self, array: np.ndarray, min_bg_tr=None, fallback_fit_results: FitResults=None):
        if min_bg_tr is None:
            min_bg_tr = self.quantile_ignore_all_na(array, self.config.min_background_prop)

        if self.config.max_background_prop <= self.config.min_background_prop:
            max_bg_tr = min_bg_tr + 1
        else:
            max_bg_tr = self.quantile_ignore_all_na(array, self.config.max_background_prop)

        if fallback_fit_results is not None:
            max_bg_tr = np.maximum(
                max_bg_tr,
                fallback_fit_results.fit_threshold
            )
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

            condition = value_counts[i] < self.config.min_obs_rmsea
            value_counts[i - 1] += np.where(
                condition,
                value_counts[i],
                0
            )
            bin_edges[i] = np.where(
                condition,
                bin_edges[i + 1],
                bin_edges[i]
            )

            value_counts[i] = np.where(
                condition,
                0,
                value_counts[i]
            )

    def quantile_ignore_all_na(self, array: np.ndarray, quantile: float):
        """
        Wrapper of nan quantile to avoid runtime warnings when data is all nan.
        """
        all_nan = np.all(np.isnan(array), axis=0)
        if array.ndim == 1:
            return np.nan if all_nan else np.nanquantile(array, quantile)
        result = np.full(array.shape[1], np.nan, dtype=np.float32)
        result[~all_nan] = np.nanquantile(array[:, ~all_nan], quantile, axis=0, method='higher')
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
        valid_bins = np.ones_like(value_counts_per_bin, dtype=bool)
        bg_sum_mappable = np.sum(value_counts_per_bin, axis=0, where=valid_bins)

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
        return calc_rmsea(
            value_counts_per_bin,
            expected_counts,
            bg_sum_mappable,
            df,
            where=valid_bins
        )

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
            fallback_fit_results (FitResults): Fallback fit results to use if the fit has failed.
        """

        data_for_fit = self.prepare_data_for_fit(agg_cutcounts, step, fallback_fit_results)
        best_fit_result = self.fit_and_choose_best(data_for_fit)
        if not check_valid_nb_params(best_fit_result):
            if fallback_fit_results is not None:
                best_fit_result = self.fit_and_choose_best(data_for_fit, fallback_fit_results)
            else:
                raise NotEnoughDataForContig
        best_fit_result.n_total = agg_cutcounts.count()
        signal_mask = self.get_signal_mask_for_tr(
            agg_cutcounts,
            best_fit_result.fit_threshold
        )
        best_fit_result.n_signal = signal_mask.sum()

        best_fit_result.signal_tags = agg_cutcounts[signal_mask].compressed().astype(np.int64).sum()
       
        best_fit_result.total_tags = np.nansum(agg_cutcounts[np.isfinite(agg_cutcounts)].compressed().astype(np.int64))

        return best_fit_result
    
    def fit_and_choose_best(self, data_for_fit: DataForFit, fallback_fit_results: FitResults=None):
        print('No warning')
        result = self.fit_all_thresholds(data_for_fit, fallback_fit_results)
        print('No warning?')
        if len(result) == 0: # No valid fits => fit all data
            best_fit_result = self.fit_for_tr(
                data_for_fit,
                np.inf,
                fallback_fit_results=fallback_fit_results,
                calc_rmsea=False
            )
        else:
            best_fit_result = min(result, key=lambda x: x.rmsea)
        return best_fit_result
    
    def prepare_data_for_fit(
            self,
            agg_cutcounts: ma.MaskedArray,
            step: int=None,
            fallback_fit_results: FitResults=None
        ):
        if step is None:
            step = self.config.window
        agg_cutcounts = agg_cutcounts.filled(np.nan)
        max_counts_with_flanks = self.get_max_count_with_flanks(agg_cutcounts)[::step]
        agg_cutcounts = agg_cutcounts[::step]
        bin_edges, n_signal_bins = self.get_all_bins(agg_cutcounts, fallback_fit_results)
        bin_edges = bin_edges[:, None]
        value_counts = self.value_counts_per_bin(agg_cutcounts[:, None], bin_edges)
        self.merge_signal_bins_for_rmsea_inplace(value_counts, bin_edges, n_signal_bins)
        return DataForFit(
            bin_edges,
            value_counts,
            n_signal_bins,
            agg_cutcounts,
            max_counts_with_flanks
        )
    
    def fit_all_thresholds(
            self,
            data_for_fit: DataForFit,
            fallback_fit_results: FitResults=None
            ) -> List[FitResults]:
        result = []
        for i in np.arange(data_for_fit.n_signal_bins)[::-1]:
            tr = data_for_fit.bin_edges[len(data_for_fit.bin_edges) - i - 1, 0]
            try:
                assumed_signal_mask = data_for_fit.max_counts_with_flanks >= tr
                step_fit = self._fit_for_bg_mask(
                    data_for_fit.agg_cutcounts,
                    bin_edges=data_for_fit.bin_edges,
                    assumed_signal_mask=assumed_signal_mask,
                    fallback_fit_results=fallback_fit_results
                )
            except NotEnoughDataForContig:
                continue
            q = self.get_bg_quantile_from_tr(data_for_fit.agg_cutcounts, tr)
            step_fit.fit_quantile = q
            step_fit.fit_threshold = tr
            result.append(step_fit)
        return result
    
    @wrap_masked
    def fit_for_tr(
            self,
            data_for_fit: DataForFit,
            tr: float,
            fallback_fit_results: FitResults=None,
            calc_rmsea=True
    ):
        assumed_signal_mask = data_for_fit.max_counts_with_flanks >= tr

        step_fit = self._fit_for_bg_mask(
            data_for_fit.agg_cutcounts,
            assumed_signal_mask,
            data_for_fit.bin_edges,
            fallback_fit_results=fallback_fit_results,
            calc_rmsea=calc_rmsea
        )
        step_fit.fit_quantile = self.get_bg_quantile_from_tr(data_for_fit.agg_cutcounts, tr)
        step_fit.fit_threshold = tr
        return step_fit
    
    def _fit_for_bg_mask(
        self,
        agg_cutcounts,
        assumed_signal_mask,
        bin_edges,
        fallback_fit_results: FitResults=None,
        calc_rmsea=True
    ):
        mean, var = self.get_mean_and_var(agg_cutcounts, where=~assumed_signal_mask)
        if fallback_fit_results is not None:
            r = fallback_fit_results.r
            n_params = 1
        else:
            r = self.r_from_mean_and_var(mean, var)
            n_params = 2
        p = self.p_from_mean_and_r(mean, r)

        if not check_valid_nb_params(FitResults(p, r)):
            raise NotEnoughDataForContig
        
        if calc_rmsea: # Used to bypass RMSEA calculation
            # in the code used when np.inf passed 

            value_counts = self.value_counts_per_bin(
                agg_cutcounts[:, None],
                bin_edges,
                where=~assumed_signal_mask[:, None]
            )
            rmsea = self.calc_rmsea_all_windows(
                p,
                r,
                n_params,
                bin_edges,
                value_counts,
            )[0]
            
            if not np.isfinite(rmsea):
                raise NotEnoughDataForContig
        else:
            rmsea = np.nan
        return FitResults(p, r, rmsea)


class StridedBackgroundFit(BackgroundFit):
    """
    Class to fit the background distribution using a strided approach.
    Extemely computationally taxing, use large stride values to compensate.
    """

    def find_thresholds_at_chrom_quantile(
            self,
            agg_cutcounts: np.ndarray,
            fit_quantile: float,
            bg_window=None
        ):
        original_shape = agg_cutcounts.shape
        strided_agg_cutcounts = self.get_strided_agg_cutcounts(agg_cutcounts, bg_window=bg_window)
        subsampled_indices = self.get_subsampled_indices(original_shape)
        values = self.quantile_ignore_all_na(
            strided_agg_cutcounts,
            fit_quantile
        )
        return self.upcast(original_shape, subsampled_indices, values)

    def get_strided_agg_cutcounts(self, agg_cutcounts: np.ndarray, bg_window=None):
        if bg_window is None:
            bg_window = self.config.bg_window
        points_in_bg_window = (bg_window - 1) // self.sampling_step
        if points_in_bg_window % 2 == 0:
            points_in_bg_window += 1
        
        strided_agg_cutcounts = rolling_view_with_nan_padding(
            agg_cutcounts[::self.sampling_step],
            points_in_window=points_in_bg_window,
            interpolation_step=self.interpolation_step
        ) # shape (bg_window, n_points)

        return strided_agg_cutcounts
    
    def get_subsampled_indices(self, original_shape: tuple):
        return np.arange(0, original_shape[0], self.sampling_step, dtype=np.uint32)[::self.interpolation_step]
    
    def upcast(self, original_shape, subsampled_indices: np.ndarray, values: np.ndarray):
        upcasted = np.full(original_shape, np.nan, dtype=np.float16)
        upcasted[subsampled_indices] = np.asarray(values, dtype=np.float16)
        upcasted[~np.isfinite(upcasted)] = np.nan
        return upcasted


class WindowBackgroundFit(BackgroundFit):
    """
    Class to fit the background distribution in a running window fashion
    """
    def fit(self, array: ma.MaskedArray, per_window_trs, fallback_fit_results: FitResults=None) -> WindowedFitResults:
        agg_cutcounts = array.copy()

        high_signal_mask = self.get_signal_mask_for_tr(
            agg_cutcounts,
            per_window_trs,
            flank_length=self.config.exclude_peak_flank_scoring
        )
        agg_cutcounts[high_signal_mask] = np.nan
        fit_results = self.sliding_method_of_moments_fit(
            agg_cutcounts,
            fallback_fit_results=fallback_fit_results
        )

        return fit_results
    
    def sliding_method_of_moments_fit(
            self,
            agg_cutcounts: ma.MaskedArray,
            fallback_fit_results: FitResults
        ):

        bg_window = self.config.bg_window
        mean = self.centered_running_nanmean(agg_cutcounts, bg_window)
        
        mean = ma.masked_invalid(mean)
        enough_bg_mask = ~mean.mask
        
        if fallback_fit_results is not None:
            r = np.full_like(mean, fallback_fit_results.r, dtype=np.float32)
        else:
            var = self.centered_running_nanvar(agg_cutcounts, bg_window)
            var = ma.masked_invalid(var)
            r = self.r_from_mean_and_var(mean, var).filled(np.nan)

        p = self.p_from_mean_and_r(mean, r).filled(np.nan)
        return WindowedFitResults(p, r, enough_bg_mask=enough_bg_mask)
