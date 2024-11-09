import bottleneck as bn
import numpy as np

from hotspot3.io.logging import WithLogger
from hotspot3.utils import wrap_masked, correct_offset

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
    @correct_offset
    def centered_running_nanmax(self, array, window):
        min_count = self.get_min_count(window)
        return bn.move_max(array, window, min_count=1).astype(np.float32)
    
    @wrap_masked
    def running_nanmedian(self, array, window):
        return bn.move_median(array, window).astype(np.float32)

    def get_min_count(self, window):
        return max(1, round(window * self.config.min_mappable_bg_frac))
    
    @wrap_masked
    def get_max_count_with_flanks(self, array: np.ndarray):
        flanks_window = self.config.exclude_peak_flank_length * 2 + 1
        return self.centered_running_nanmax(array, flanks_window)
    
    def get_signal_mask_for_tr(self, array: np.ndarray, tr: float):
        max_count = self.get_max_count_with_flanks(array)
        return max_count >= tr