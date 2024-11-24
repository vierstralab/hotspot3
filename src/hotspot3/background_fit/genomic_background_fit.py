
import numpy.ma as ma
import numpy as np
import pandas as pd
from typing import List

from genome_tools.genomic_interval import GenomicInterval, genomic_intervals_to_df

from hotspot3.io.logging import WithLoggerAndInterval
from hotspot3.background_fit.fit import GlobalBackgroundFit, StridedBackgroundFit, WindowBackgroundFit
from hotspot3.helpers.models import FitResults, WindowedFitResults, NotEnoughDataForContig
from hotspot3.helpers.format_converters import convert_fit_results_to_series
from hotspot3.background_fit import check_valid_nb_params
from hotspot3.helpers.utils import interpolate_nan


class SegmentalFit(WithLoggerAndInterval):
    def filter_signal_to_segment(self, agg_cutcounts: ma.MaskedArray, segment: GenomicInterval=None) -> ma.MaskedArray:
        if segment is None:
            segment = self.genomic_interval
        return agg_cutcounts[segment.start:segment.end]
    
    def fit_segments(
            self,
            agg_cutcounts: ma.MaskedArray,
            bad_segments: List[GenomicInterval],
            fallback_fit_results: FitResults,
            min_bg_tag_proportion: float=None
        ):
        total_len = agg_cutcounts.count()
        fit_res = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float16)
        windowed_fit_results = WindowedFitResults(
            p=fit_res,
            r=fit_res.copy(),
            enough_bg_mask=np.zeros(agg_cutcounts.shape[0], dtype=bool)
        )
        per_window_trs = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float16)

        intervals_stats = genomic_intervals_to_df(bad_segments).drop(columns=['chrom', 'name'])
        for i, segment_interval in enumerate(bad_segments):
            start = int(segment_interval.start)
            end = int(segment_interval.end)        
            signal_at_segment = self.filter_signal_to_segment(
                agg_cutcounts,
                segment_interval
            )
            g_fit = self.copy_with_params(
                GlobalBackgroundFit,
                name=segment_interval.to_ucsc()
            )
            try:
                if min_bg_tag_proportion is not None:
                    uq, cts = np.unique(signal_at_segment, return_counts=True)
                    total = uq * cts
                    valid_cts = uq[np.cumsum(total) / (total).sum() >= min_bg_tag_proportion]
                    if valid_cts.size == 0:
                        self.logger.critical(f"{segment_interval.to_ucsc()}: Not enough background data")
                        raise ValueError
                    min_quantile = g_fit.get_bg_quantile_from_tr(signal_at_segment, valid_cts[0])
                    g_fit.config.min_background_prop = min_quantile

                segment_fit_results = g_fit.fit(
                    signal_at_segment,
                    step=self.get_optimal_segment_step(total_len, len(segment_interval)),
                    fallback_fit_results=fallback_fit_results
                )
                success_fit = True

                fit_res = self.fit_segment_params(
                    signal_at_segment,
                    segment_fit_results.fit_threshold,
                    fallback_fit_results=segment_fit_results,
                    genomic_interval=segment_interval
                )
            except NotEnoughDataForContig:
                segment_fit_results = FitResults()
                fit_res = WindowedFitResults(
                    p=np.full(signal_at_segment.shape[0], np.nan, dtype=np.float16),
                    r=np.full(signal_at_segment.shape[0], np.nan, dtype=np.float16),
                    enough_bg_mask=np.zeros(signal_at_segment.shape[0], dtype=bool)
                )
                success_fit = False
            
            # TODO wrap in a function
            fit_series = convert_fit_results_to_series(
                segment_fit_results,
                signal_at_segment.compressed().mean(),
                fit_type='segment',
                success_fit=success_fit
            )
            self.set_dtype(intervals_stats, fit_series)
            intervals_stats.loc[i, fit_series.index] = fit_series
            ###

            windowed_fit_results.r[start:end] = fit_res.r
            windowed_fit_results.p[start:end] = fit_res.p
            windowed_fit_results.enough_bg_mask[start:end] = fit_res.enough_bg_mask
            per_window_trs[start:end] = segment_fit_results.fit_threshold
        
        return windowed_fit_results, per_window_trs, intervals_stats


    def set_dtype(self, intervals_stats: pd.DataFrame, fit_series: pd.Series):
        """
        Workaround func to avoid future warnings about setting bool values to float (default) columns
        Initially sets value of first row to every row in the DataFrame
        """
        added_cols = ~fit_series.index.isin(intervals_stats.columns)
        if np.any(added_cols):
            for col in fit_series.index[added_cols]:
                intervals_stats[col] = fit_series[col]

    def add_fallback_fit_stats(
            self,
            agg_cutcounts: ma.MaskedArray,
            fallback_fit_results: FitResults,
            segment_stats: pd.DataFrame
        ):
        intervals_stats = genomic_intervals_to_df([self.genomic_interval]).drop(
            columns=['chrom', 'name']
        )
        intervals_stats['BAD'] = 0
        fit_series = convert_fit_results_to_series(
            fallback_fit_results,
            agg_cutcounts.compressed().mean(),
            fit_type='global',
            success_fit=True,
        )
        self.set_dtype(intervals_stats, fit_series)
        intervals_stats.loc[0, fit_series.index] = fit_series
        return pd.concat([intervals_stats, segment_stats], ignore_index=True)


    def get_optimal_segment_step(self, total_len, segment_len):
        # Scale step according to the length of segment
        return max(
            20, 
            round(self.config.chromosome_fit_step / total_len * segment_len)
        )

    def fit_segment_params(
            self,
            signal_at_segment: ma.MaskedArray,
            thresholds: np.ndarray,
            fallback_fit_results: FitResults,
            genomic_interval: GenomicInterval=None
        ) -> WindowedFitResults:
        if genomic_interval is None:
            genomic_interval = self.genomic_interval

        w_fit = self.copy_with_params(
            WindowBackgroundFit,
            name=genomic_interval.to_ucsc()
        )

        fit_res = w_fit.fit(signal_at_segment, per_window_trs=thresholds)
        success_fits = check_valid_nb_params(fit_res) & fit_res.enough_bg_mask

        need_global_fit = ~success_fits & fit_res.enough_bg_mask
        if np.any(need_global_fit):
            fit_res.r[need_global_fit] = fallback_fit_results.r

            fit_res.p[need_global_fit] = w_fit.fit(
                signal_at_segment,
                per_window_trs=thresholds,
                fallback_fit_results=fallback_fit_results
            ).p[need_global_fit]
        self.logger.debug(f"{genomic_interval.to_ucsc()}: Successfully fit per-bp negative-binomial model for {np.sum(success_fits):,} bp. Use segment r for {np.sum(need_global_fit):,} bp")
        return fit_res


class ChromosomeFit(WithLoggerAndInterval):
    
    def fit_segment_thresholds(
            self,
            agg_cutcounts: ma.MaskedArray,
            step=None
        ):
        if step is None:
            step = self.config.chromosome_fit_step

        g_fit = self.copy_with_params(GlobalBackgroundFit)
        global_fit_results = g_fit.fit(agg_cutcounts, step=step)
        if not check_valid_nb_params(global_fit_results):
            raise NotEnoughDataForContig
        strided_fit = self.copy_with_params(StridedBackgroundFit)
        fit_threshold = strided_fit.find_thresholds_at_chrom_quantile(
            agg_cutcounts,
            global_fit_results.fit_quantile,
            bg_window=self.config.bg_window
        )
        fit_threshold2 = strided_fit.find_thresholds_at_chrom_quantile(
            agg_cutcounts,
            global_fit_results.fit_quantile,
            bg_window=self.config.bg_window_small
        )
        fit_threshold = interpolate_nan(np.minimum(fit_threshold, fit_threshold2))
        self.logger.debug(f"{self.name}: Signal thresholds approximated")
        return fit_threshold, global_fit_results
