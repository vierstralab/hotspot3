from hotspot3.logging import WithLogger
from hotspot3.fit import GlobalBackgroundFit, StridedBackgroundFit, WindowBackgroundFit
from hotspot3.models import GlobalFitResults, WindowedFitResults
from hotspot3.utils import interpolate_nan
from hotspot3.stats import check_valid_fit
import numpy.ma as ma
import numpy as np
from genome_tools.genomic_interval import GenomicInterval


class SegmentFit(WithLogger):
    def __init__(self, genomic_interval: GenomicInterval, config=None, logger=None):
        name = self.genomic_interval
        super().__init__(logger=logger, config=config, name=name)
        self.genomic_interval = genomic_interval
    
    def filter_signal_to_segment(self, agg_cutcounts: np.ndarray) -> np.ndarray:
        return agg_cutcounts[self.genomic_interval.start:self.genomic_interval.end]
    
    def fit_segment_thresholds(self, agg_cutcounts: ma.MaskedArray, global_fit: GlobalFitResults=None):
        signal_at_segment = self.filter_signal_to_segment(agg_cutcounts)
        s_fit = GlobalBackgroundFit(self.config, name=self.name)
        segment_fit = s_fit.fit(signal_at_segment)
        valid_segment = check_valid_fit(segment_fit)
        if not valid_segment:
            segment_fit = s_fit.fit(signal_at_segment, global_fit=global_fit)

        fine_signal_level_fit = StridedBackgroundFit(self.config, name=self.name)
        thresholds, _, rmsea = fine_signal_level_fit.fit_tr(
            signal_at_segment,
            global_fit=segment_fit,
        )
        thresholds = interpolate_nan(thresholds)
        self.logger.debug(f"{self.genomic_interval}: Signal thresholds approximated")
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
        self.logger.debug(f"{self.genomic_interval}: Fit per-bp negative-binomial model for {np.sum(success_fits):,}. Use global fit for {np.sum(need_global_fit):,} windows")
        return fit_res