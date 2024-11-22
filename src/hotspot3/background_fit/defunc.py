from hotspot3.background_fit.fit import StridedBackgroundFit as StridedFit
from hotspot3.models import FitResults, WindowedFitResults, DataForFit
from hotspot3.utils import wrap_masked
import numpy as np
import numpy.ma as ma
import dataclasses

class StridedBackgroundFit(StridedFit):

    ## Currently deprecated methods ##
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

    def cast_to_original_shape(
            self,
            fit_results: FitResults,
            original_shape: tuple
        ) -> FitResults:
        subsampled_indices = self.get_subsampled_indices(original_shape)

        for data_field in dataclasses.fields(fit_results):
            field = data_field.name
            values = getattr(fit_results, field)
            if values is not None and not (np.ndim(values) == 0 and np.isnan(values)):
                upcasted = self.upcast(original_shape, subsampled_indices, values)
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

            data_for_current_threshold, assumed_signal_mask = self.get_data_for_current_threshold(
                data_for_fit,
                current_index,
                changing_indices
            )
            
            # 2. Fit for the current threshold and evaluate the fit
            bin_fit_results = self.fit_for_bin(
                data_for_current_threshold.agg_cutcounts,
                where=~assumed_signal_mask,
                fallback_fit_results=fallback_fit_results
            )
            enough_bg_mask = bin_fit_results.enough_bg_mask


            rmsea = self.evaluate_fit_for_bin(
                data_for_current_threshold,
                bin_fit_results,
                fallback_fit_results,
            )
            bin_fit_results = FitResults(
                bin_fit_results.p,
                bin_fit_results.r,
                rmsea=rmsea,
                fit_threshold=data_for_current_threshold.bin_edges[-1]
            )

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
        
        best_fit_results.fit_quantile = self.get_bg_quantile_from_tr(
            data_for_fit.agg_cutcounts,
            best_fit_results.fit_threshold
        )
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
        currect_counts_with_flanks = data_for_fit.max_counts_with_flanks[:, changing_indices]
        
        assumed_signal_mask = currect_counts_with_flanks >= current_bin_edges[-1]

        # TODO: don't update the value counts from first iteration
        current_value_counts = self.value_counts_per_bin(
            current_agg_cutcounts,
            current_bin_edges,
            where=~assumed_signal_mask
        )

        return DataForFit(
            current_bin_edges,
            current_value_counts,
            data_for_fit.n_signal_bins,
            current_agg_cutcounts,
            currect_counts_with_flanks
        ), assumed_signal_mask


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
                self.quantile_ignore_all_na(data_for_fit.agg_cutcounts, self.config.min_background_prop) 
            )
        )