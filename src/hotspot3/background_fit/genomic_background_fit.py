
import numpy.ma as ma
import numpy as np
import pandas as pd
from typing import List

from genome_tools.genomic_interval import GenomicInterval, genomic_intervals_to_df, df_to_genomic_intervals

from hotspot3.io.logging import WithLoggerAndInterval
from hotspot3.background_fit.fit import GlobalBackgroundFit, StridedBackgroundFit, WindowBackgroundFit
from hotspot3.helpers.models import FitResults, WindowedFitResults, NotEnoughDataForContig
from hotspot3.helpers.format_converters import set_series_row_to_df, convert_fit_results_to_series, fit_stats_df_to_fallback_fit_results
from hotspot3.helpers.stats import check_valid_nb_params, threhold_from_bg_tag_proportion
from hotspot3.helpers.utils import interpolate_nan


class SegmentalFit(WithLoggerAndInterval):
    def filter_signal_to_segment(self, agg_cutcounts: ma.MaskedArray, segment: GenomicInterval=None) -> ma.MaskedArray:
        if segment is None:
            segment = self.genomic_interval
        return agg_cutcounts[segment.start:segment.end]
    
    def per_bp_background_model_fit(
            self,
            agg_cutcounts: ma.MaskedArray,
            segments: pd.DataFrame,
        ):
        per_window_trs = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float16)
        windowed_fit_results = WindowedFitResults(
            p=per_window_trs.copy(),
            r=per_window_trs.copy(),
            enough_bg_mask=np.zeros(agg_cutcounts.shape[0], dtype=bool)
        )
        segments_genomic_intervals = df_to_genomic_intervals(segments)

        for i, segment_interval in enumerate(segments_genomic_intervals):
            start = int(segment_interval.start)
            end = int(segment_interval.end)
            segment_fit_result = fit_stats_df_to_fallback_fit_results(segments.iloc[[i]])
            if not check_valid_nb_params(segment_fit_result):
                continue
            signal_at_segment = self.filter_signal_to_segment(agg_cutcounts, segment_interval)
            fit_res = self.fit_segment_params(
                signal_at_segment,
                segment_fit_result.fit_threshold,
                fallback_fit_results=segment_fit_result,
                genomic_interval=segment_interval
            )
            windowed_fit_results.r[start:end] = fit_res.r
            windowed_fit_results.p[start:end] = fit_res.p
            windowed_fit_results.enough_bg_mask[start:end] = fit_res.enough_bg_mask
            per_window_trs[start:end] = segment_fit_result.fit_threshold
        return windowed_fit_results, per_window_trs
    
    def fit_per_segment_bg_model(
            self,
            agg_cutcounts: ma.MaskedArray,
            bad_segments: List[GenomicInterval],
            fallback_fit_results: FitResults,
            min_bg_tag_proportion: np.ndarray=None
        ):
        total_len = agg_cutcounts.count()
        intervals_stats = genomic_intervals_to_df(bad_segments).drop(columns=['chrom', 'name'])
        if min_bg_tag_proportion is not None:
            assert len(min_bg_tag_proportion) == len(bad_segments), "min_bg_tag_proportion should have the same length as segments"
        self.logger.debug(f"{self.genomic_interval.chrom}: Fitting background model for {len(bad_segments)} segments")
        for i, segment_interval in enumerate(bad_segments):
            signal_at_segment = self.filter_signal_to_segment(agg_cutcounts, segment_interval)
            segment_mean_signal = np.mean(signal_at_segment.compressed())
            g_fit = self.copy_with_params(
                GlobalBackgroundFit,
                name=segment_interval.to_ucsc()
            )
            try:
                step = self.get_optimal_segment_step(total_len, len(segment_interval))
                if min_bg_tag_proportion is not None:
                    signal_with_step = signal_at_segment[::step].compressed()
                    valid_count = threhold_from_bg_tag_proportion(
                        signal_with_step,
                        min_bg_tag_proportion[i],
                    )
                    min_bg_quantile = g_fit.get_bg_quantile_from_tr(signal_with_step, valid_count)
                    g_fit.config.min_background_prop = min(
                        g_fit.config.max_background_prop,
                        min_bg_quantile
                    )

                segment_fit_results = g_fit.fit(
                    signal_at_segment,
                    step=step,
                    fallback_fit_results=fallback_fit_results
                )
                success_fit = True

            except NotEnoughDataForContig:
                segment_fit_results = FitResults()
                success_fit = False
            
            fit_series = convert_fit_results_to_series(
                segment_fit_results,
                fit_type='segment',
                success_fit=success_fit
            )
            set_series_row_to_df(intervals_stats, fit_series, i)
        return intervals_stats


    def add_fallback_fit_stats(
            self,
            fallback_fit_results: FitResults,
            segment_stats: pd.DataFrame
        ):
        intervals_stats = genomic_intervals_to_df([self.genomic_interval]).drop(
            columns=['chrom', 'name']
        )
        intervals_stats['BAD'] = 0
        fit_series = convert_fit_results_to_series(
            fallback_fit_results,
            fit_type='global',
            success_fit=True,
        )
        set_series_row_to_df(intervals_stats, fit_series, 0)
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
        self.logger.debug(f"{genomic_interval.to_ucsc()}: Successfully fit background model for {np.sum(success_fits):,} bp. Use segment r for {np.sum(need_global_fit):,} bp")
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
