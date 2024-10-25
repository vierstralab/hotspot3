import numpy.ma as ma
import numpy as np
from scipy import stats as st
from hotspot3.models import NoContigPresentError, ProcessorConfig, FitResults
from hotspot3.utils import wrap_masked
import bottleneck as bn


class BackgroundFit:
    def __init__(self, config: ProcessorConfig=None):
        if config is None:
            config = ProcessorConfig()
        self.config = config

    def fit(self) -> FitResults:
        raise NotImplementedError

    def p_from_mean_and_var(self, mean: np.ndarray, var: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            p = np.asarray(1 - mean / var, dtype=np.float32)
        return p
    
    def r_from_mean_and_var(self, mean: np.ndarray, var: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.asarray(mean ** 2 / (var - mean), dtype=np.float32)
        return r

    def calc_rmsea_for_tr(self, obs, unique_cutcounts, r, p, tr):
        N = sum(obs)
        exp = st.nbinom.pmf(unique_cutcounts, r, 1 - p) / st.nbinom.cdf(tr - 1, r, 1 - p) * N
        # chisq = sum((obs - exp) ** 2 / exp)
        G_sq = 2 * sum(obs * np.log(obs / exp))
        df = len(obs) - 2
        return np.sqrt((G_sq / df - 1) / (N - 1))


class GlobalBackgroundFit(BackgroundFit):
    def fit(self, agg_cutcounts: ma.MaskedArray, tr: int) -> FitResults:
        high_signal_mask = agg_cutcounts > tr
        data = agg_cutcounts[~high_signal_mask].compressed()
        mean, var = self.estimate_global_mean_and_var(data)
        p = self.p_from_mean_and_var(mean, var)
        r = self.r_from_mean_and_var(mean, var)
        unique, counts = np.unique(data, return_counts=True)
        rmsea = self.calc_rmsea_for_tr(counts, unique, r, p, tr)
        return FitResults(mean, var, p, r, rmsea)
        

    def estimate_global_mean_and_var(self, agg_cutcounts:np.ndarray):
        has_enough_background = np.count_nonzero(agg_cutcounts) / agg_cutcounts.size > self.config.nonzero_windows_to_fit
        if not has_enough_background:
            raise NoContigPresentError
        
        mean = np.mean(agg_cutcounts)
        variance = np.var(agg_cutcounts, ddof=1)
        return mean, variance


class WindowBackgroundFit(BackgroundFit):
    def fit(self, agg_cutcounts, tr) -> FitResults:
        high_signal_mask = agg_cutcounts > tr
        mask = agg_cutcounts.mask

        data = agg_cutcounts.filled(np.nan)
        data[high_signal_mask] = np.nan
        mean, var = self.sliding_mean_and_variance(data)
        p, r = self.p_and_r_from_mean_and_var(mean, var)
        

    def sliding_mean_and_variance(self, array): # FIXME bottleneck returns dtype float64
        window = self.config.bg_window

        mean = self.running_nanmean(array, window)
        var = bn.move_var(array, window, ddof=1, min_count=self.config.min_mappable_bg)
        return mean, var
    
    def running_nanmean(self, array, window):
        return bn.move_mean(array, window, min_count=self.config.min_mappable_bg).astype(np.float32)
    
    def sliding_r(self, mean: ma.MaskedArray, var: ma.MaskedArray) -> ma.MaskedArray:
        return wrap_masked(self.r_from_mean_and_var, mean, var)

    def sliding_p(self, mean: ma.MaskedArray, var: ma.MaskedArray) -> ma.MaskedArray:
        return wrap_masked(self.p_from_mean_and_var, mean, var)
