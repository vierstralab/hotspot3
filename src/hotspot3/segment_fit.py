from hotspot3.logging import WithLogger
from hotspot3.fit import GlobalBackgroundFit, StridedBackgroundFit, WindowBackgroundFit
from hotspot3.models import GlobalFitResults, WindowedFitResults
from hotspot3.utils import interpolate_nan
from hotspot3.stats import check_valid_fit
import numpy.ma as ma
import numpy as np
from genome_tools.genomic_interval import GenomicInterval


class ChromosomeFit(WithLogger):
    def __init__(self, genomic_interval: GenomicInterval, config=None, logger=None):
        super().__init__(logger=logger, config=config, name=genomic_interval.chrom)
        self.genomic_interval = genomic_interval

    def fit_params(self, agg_cutcounts: ma.MaskedArray, bad_segments, global_fit: GlobalFitResults=None):
        final_r = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float32)
        final_p = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float32)
        final_rmsea = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float16)
        per_window_trs = np.full(agg_cutcounts.shape[0], np.nan, dtype=np.float16)
        enough_bg = np.zeros(agg_cutcounts.shape[0], dtype=bool)

        for segment in bad_segments:
            start = int(segment.start)
            end = int(segment.end)
            segment_interval = GenomicInterval(self.genomic_interval.chrom, start, end)
            s_fit = SegmentFit(segment_interval, self.config, logger=self.logger)
            thresholds, rmsea, global_seg_fit = s_fit.fit_segment_thresholds(
                agg_cutcounts,
                global_fit=global_fit
            )

            fit_res = s_fit.fit_segment_params(agg_cutcounts, thresholds, global_fit=global_seg_fit)

            final_r[start:end] = fit_res.r
            final_p[start:end] = fit_res.p
            final_rmsea[start:end] = rmsea
            per_window_trs[start:end] = thresholds
            enough_bg[start:end] = fit_res.enough_bg_mask

        return WindowedFitResults(
            p=final_p,
            r=final_r,
            enough_bg_mask=enough_bg
        ), per_window_trs, final_rmsea


class SegmentFit(WithLogger):
    def __init__(self, genomic_interval: GenomicInterval, config=None, logger=None):
        self.genomic_interval = genomic_interval
        super().__init__(logger=logger, config=config, name=genomic_interval.to_ucsc())
    
    def filter_signal_to_segment(self, agg_cutcounts: np.ndarray) -> np.ndarray:
        return agg_cutcounts[self.genomic_interval.start:self.genomic_interval.end]
    
    def fit_segment_thresholds(self, agg_cutcounts: ma.MaskedArray, global_fit: GlobalFitResults=None, step=None):
        signal_at_segment = self.filter_signal_to_segment(agg_cutcounts)
        g_fit = GlobalBackgroundFit(self.config, name=self.name)

        segment_fit = g_fit.fit(signal_at_segment, step=step)
        valid_segment = check_valid_fit(segment_fit)
        if not valid_segment:
            segment_fit = g_fit.fit(signal_at_segment, global_fit=global_fit, step=step)

        fine_signal_level_fit = StridedBackgroundFit(self.config, name=self.name)
        thresholds, _, rmsea = fine_signal_level_fit.fit_tr(
            signal_at_segment,
            global_fit=segment_fit,
        )
        thresholds = interpolate_nan(thresholds)
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

