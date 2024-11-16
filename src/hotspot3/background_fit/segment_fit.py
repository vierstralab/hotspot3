
import numpy.ma as ma
import numpy as np
from typing import List

from genome_tools.genomic_interval import GenomicInterval, genomic_intervals_to_df

from hotspot3.io.logging import WithLoggerAndInterval
from hotspot3.background_fit.fit import GlobalBackgroundFit, StridedBackgroundFit, WindowBackgroundFit
from hotspot3.models import FitResults, WindowedFitResults, NotEnoughDataForContig

from hotspot3.background_fit import check_valid_fit
from hotspot3.utils import interpolate_nan


class SegmentsFit(WithLoggerAndInterval):
    def filter_signal_to_segment(self, agg_cutcounts: np.ndarray, segment: GenomicInterval=None) -> np.ndarray:
        if segment is None:
            segment = self.genomic_interval
        return agg_cutcounts[segment.start:segment.end]


    def fit_params(self, agg_cutcounts: ma.MaskedArray, bad_segments: List[GenomicInterval], fallback_fit_results: FitResults=None):
        fit_res = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float32)
        windowed_fit_results = WindowedFitResults(
            p=fit_res,
            r=fit_res.copy(),
            enough_bg_mask=np.zeros(agg_cutcounts.shape[0], dtype=bool)
        )
        final_rmsea = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float16)
        per_window_trs = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float16)
        
        segment_fits: List[FitResults] = []
        segments = bad_segments
        types = []
        success_fit = []
        if fallback_fit_results is not None:
            segment_fits.append(fallback_fit_results)
            self.genomic_interval.BAD = None
            segments = [self.genomic_interval, *segments]
            types.append('global')
            success_fit.append(True)

        for segment_interval in bad_segments:
            start = int(segment_interval.start)
            end = int(segment_interval.end)        
            types.append('segment')
            segment_step = 20
            signal_at_segment = self.filter_signal_to_segment(
                agg_cutcounts,
                segment_interval
            )
            try:
                g_fit = self.copy_with_params(
                    GlobalBackgroundFit,
                    name=segment_interval.to_ucsc()
                )

                segment_fit_results = g_fit.fit(signal_at_segment, step=segment_step)
                if not check_valid_fit(segment_fit_results) and fallback_fit_results is not None:
                    segment_fit_results = g_fit.fit(
                        signal_at_segment,
                        fallback_fit_results=fallback_fit_results,
                        step=segment_step
                    )
                success_fit.append(True)
            except NotEnoughDataForContig:
                segment_fit_results = fallback_fit_results
                success_fit.append(False)
            
            segment_fits.append(segment_fit_results)

            fit_res = self.fit_segment_params(
                signal_at_segment,
                segment_fit_results.fit_threshold,
                fallback_fit_results=segment_fit_results,
                genomic_interval=segment_interval
            )

            windowed_fit_results.r[start:end] = fit_res.r
            windowed_fit_results.p[start:end] = fit_res.p
            windowed_fit_results.enough_bg_mask[start:end] = fit_res.enough_bg_mask

            final_rmsea[start:end] = segment_fit_results.rmsea
            per_window_trs[start:end] = segment_fit_results.fit_threshold

        intervals_stats = genomic_intervals_to_df(segments).drop(columns=['chrom', 'name'])
        intervals_stats['r'] = [x.r for x in segment_fits]
        intervals_stats['p'] = [x.p for x in segment_fits]
        intervals_stats['rmsea'] = [x.rmsea for x in segment_fits]
        intervals_stats['signal_tr'] = [x.fit_threshold for x in segment_fits]
        intervals_stats['quantile_tr'] = [x.fit_quantile for x in segment_fits]
        intervals_stats['fit_type'] = types
        intervals_stats['success_fit'] = success_fit

        # FIXME wrap in dataclass
        return windowed_fit_results, per_window_trs, final_rmsea, intervals_stats


    def fit_segment_params(
            self,
            signal_at_segment: ma.MaskedArray,
            thresholds: np.ndarray,
            fallback_fit_results: FitResults=None,
            genomic_interval: GenomicInterval=None
        ) -> WindowedFitResults:
        if genomic_interval is None:
            genomic_interval = self.genomic_interval

        w_fit = self.copy_with_params(
            WindowBackgroundFit,
            name=genomic_interval.to_ucsc()
        )

        fit_res = w_fit.fit(signal_at_segment, per_window_trs=thresholds)
        success_fits = check_valid_fit(fit_res) & fit_res.enough_bg_mask

        need_global_fit = ~success_fits & fit_res.enough_bg_mask
        fit_res.r[need_global_fit] = fallback_fit_results.r

        if fallback_fit_results.rmsea > self.config.rmsea_tr: 
            fit_res.p[need_global_fit] = fallback_fit_results.p
        else:
            fit_res.p[need_global_fit] = w_fit.fit(
                signal_at_segment,
                per_window_trs=thresholds,
                fallback_fit_results=fallback_fit_results
            ).p[need_global_fit]
        self.logger.debug(f"{genomic_interval.to_ucsc()}: Fit per-bp negative-binomial model for {np.sum(success_fits):,}. Use global fit for {np.sum(need_global_fit):,} windows")
        return fit_res


class ChromosomeFit(WithLoggerAndInterval):
    
    def fit_segment_thresholds(
            self,
            agg_cutcounts: ma.MaskedArray,
            step=None
        ):
        if step is None:
            step = self.config.window

        g_fit = self.copy_with_params(GlobalBackgroundFit)
        global_fit_results = g_fit.fit(agg_cutcounts, step=step)
        if not check_valid_fit(global_fit_results):
            raise NotEnoughDataForContig
        strided_fit = self.copy_with_params(StridedBackgroundFit)
        fit_threshold = strided_fit.find_thresholds_at_chrom_quantile(
            agg_cutcounts,
            global_fit_results.fit_quantile,
        )
        fit_threshold = interpolate_nan(fit_threshold)
        self.logger.debug(f"{self.name}: Signal thresholds approximated")
        return fit_threshold, global_fit_results
