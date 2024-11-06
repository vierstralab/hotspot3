from hotspot3.logging import WithLogger
from hotspot3.utils import wrap_masked, correct_offset
import bottleneck as bn
import numpy as np


class BottleneckWrapper(WithLogger):

    @wrap_masked
    @correct_offset
    def centered_running_nansum(self, array: np.ndarray, window: int):
        min_count = self.get_min_count(window)
        return bn.move_sum(array, window, min_count=min_count).astype(np.float32)

    @wrap_masked
    @correct_offset
    def centered_running_nanvar(self, array, window):
        min_count = self.get_min_count(window)
        return bn.move_var(array, window, ddof=1, min_count=min_count).astype(np.float32)

    @wrap_masked
    @correct_offset
    def centered_running_nanmean(self, array, window):
        min_count = self.get_min_count(window)
        return bn.move_mean(array, window, min_count=min_count).astype(np.float32)
    
    @wrap_masked
    def running_nanmedian(self, array, window):
        return bn.move_median(array, window).astype(np.float32)

    def get_min_count(self, window):
        return max(1, round(window * self.config.min_mappable_bg_frac))
    
    @wrap_masked
    def filter_by_tr_spatially(self, array: np.ndarray, tr: float):
        flanks_window = self.config.exclude_peak_flank_length * 2 + 1

        assumed_signal_mask = (array >= tr).astype(np.float32)
        assumed_signal_mask = self.centered_running_nansum(assumed_signal_mask, flanks_window) > 0
        return assumed_signal_mask