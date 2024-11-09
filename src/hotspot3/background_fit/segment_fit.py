
import numpy.ma as ma
import numpy as np
from typing import List

from genome_tools.genomic_interval import GenomicInterval, genomic_intervals_to_df

from hotspot3.io.logging import WithLoggerAndInterval
from hotspot3.background_fit.fit import GlobalBackgroundFit, StridedBackgroundFit, WindowBackgroundFit
from hotspot3.models import GlobalFitResults, WindowedFitResults, NotEnoughDataForContig

from hotspot3.utils import interpolate_nan
from hotspot3.background_fit import check_valid_fit



class ChromosomeFit(WithLoggerAndInterval):
    def fit_params(self, agg_cutcounts: ma.MaskedArray, bad_segments: List[GenomicInterval], global_fit: GlobalFitResults=None):
        final_r = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float32)
        final_p = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float32)
        final_rmsea = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float16)
        per_window_trs = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float16)
        enough_bg = np.zeros(agg_cutcounts.shape[0], dtype=bool)
        
        segment_fits: List[GlobalFitResults] = []
        segments = bad_segments
        types = []
        success_fit = []
        if global_fit is not None:
            segment_fits.append(global_fit)
            self.genomic_interval.BAD = None
            segments = [self.genomic_interval, *segments]
            types.append('global')
            success_fit.append(True)
        for segment_interval in bad_segments:
            start = int(segment_interval.start)
            end = int(segment_interval.end)
            s_fit = SegmentFit(segment_interval, self.config, logger=self.logger)            
            types.append('segment')
            try:
                thresholds, rmsea, segment_fit_results = s_fit.fit_segment_thresholds(
                    agg_cutcounts,
                    global_fit=global_fit,
                )
                success_fit.append(True)
            except NotEnoughDataForContig:
                thresholds = np.max(agg_cutcounts)
                rmsea = np.nan
                segment_fit_results = global_fit
                success_fit.append(False)
            
            segment_fits.append(segment_fit_results)

            #thresholds = np.full_like(thresholds, segment_fit_results.fit_threshold)

            fit_res = s_fit.fit_segment_params(
                agg_cutcounts,
                thresholds,
                global_fit=segment_fit_results
            )

            final_r[start:end] = fit_res.r
            final_p[start:end] = fit_res.p
            final_rmsea[start:end] = rmsea
            per_window_trs[start:end] = thresholds
            enough_bg[start:end] = fit_res.enough_bg_mask
        
        intervals_stats = genomic_intervals_to_df(segments).drop(columns=['chrom', 'name'])
        intervals_stats['r'] = [x.r for x in segment_fits]
        intervals_stats['p'] = [x.p for x in segment_fits]
        intervals_stats['rmsea'] = [x.rmsea for x in segment_fits]
        intervals_stats['signal_tr'] = [x.fit_threshold for x in segment_fits]
        intervals_stats['quantile_tr'] = [x.fit_quantile for x in segment_fits]
        intervals_stats['fit_type'] = types
        intervals_stats['success_fit'] = success_fit

        return WindowedFitResults(
            p=final_p,
            r=final_r,
            enough_bg_mask=enough_bg
        ), per_window_trs, final_rmsea, intervals_stats


class SegmentFit(WithLoggerAndInterval):
    def filter_signal_to_segment(self, agg_cutcounts: np.ndarray) -> np.ndarray:
        return agg_cutcounts[self.genomic_interval.start:self.genomic_interval.end]
    
    def fit_segment_thresholds(self, agg_cutcounts: ma.MaskedArray, global_fit: GlobalFitResults=None, step=None):
        if step is None:
            step = self.config.window

        signal_at_segment = self.filter_signal_to_segment(agg_cutcounts)
        g_fit = GlobalBackgroundFit(self.config, name=self.name)

        segment_fit = g_fit.fit(signal_at_segment, step=step)
        valid_segment = check_valid_fit(segment_fit)
        if not valid_segment and global_fit is not None:
            segment_fit = g_fit.fit(signal_at_segment, global_fit=global_fit, step=step)

        fine_signal_level_fit = StridedBackgroundFit(config=self.config, name=self.name)
        thresholds, _, rmsea = fine_signal_level_fit.fit_tr(
            signal_at_segment,
           # global_fit=segment_fit,
        )
        thresholds = interpolate_nan(thresholds).astype(np.float16)
        self.logger.debug(f"{self.name}: Signal thresholds approximated")
        return thresholds, rmsea, segment_fit
    
    def fit_segment_params(self, agg_cutcounts: ma.MaskedArray, thresholds: np.ndarray, global_fit: GlobalFitResults=None) -> WindowedFitResults:
        w_fit = WindowBackgroundFit(self.config)
        signal_at_segment = self.filter_signal_to_segment(agg_cutcounts)
        fit_res = w_fit.fit(signal_at_segment, per_window_trs=thresholds)
        success_fits = check_valid_fit(fit_res) & fit_res.enough_bg_mask
        need_global_fit = ~success_fits & fit_res.enough_bg_mask
        
        fit_res.r[need_global_fit] = global_fit.r

        if global_fit.rmsea > self.config.rmsea_tr: 
            fit_res.p[need_global_fit] = global_fit.p
        else:
            fit_res.p[need_global_fit] = w_fit.fit(
                signal_at_segment,
                per_window_trs=thresholds,
                global_fit=global_fit
            ).p[need_global_fit]
        self.logger.debug(f"{self.name}: Fit per-bp negative-binomial model for {np.sum(success_fits):,}. Use global fit for {np.sum(need_global_fit):,} windows")
        return fit_res
