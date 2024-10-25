import numpy.ma as ma
import numpy as np
from scipy import stats as st
from hotspot3.models import NoContigPresentError, ProcessorConfig, FitResults


class BackgroundFit:
    """
    Class to fit background model

    Contains methods to get negbin parameters from cutcounts and to calculate RMSEA
    """
    def __init__(self, config: ProcessorConfig=None):
        if config is None:
            config = ProcessorConfig()
        self.config = config

    def fit(self) -> FitResults:
        raise NotImplementedError

    def fit_windowed_model(self):
        ...
    
    def p_and_r_from_mean_and_var(self, mean: np.ndarray, var: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = ma.asarray(mean ** 2 / (var - mean), dtype=np.float32)
            p = ma.asarray(1 - mean / var, dtype=np.float32)
        return p, r

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
        data = agg_cutcounts[~high_signal_mask]
        mean, var = self.estimate_global_mean_and_var(data)
        p, r = self.p_and_r_from_mean_and_var(mean, var)
        rmsea = self.calc_rmsea_for_tr(data, r, p, tr)
        return FitResults(mean, var, p, r, rmsea)
        

    def estimate_global_mean_and_var(self, agg_cutcounts: ma.MaskedArray):
        has_enough_background = ma.sum(agg_cutcounts > 0) / agg_cutcounts.count() > self.config.nonzero_windows_to_fit
        if not has_enough_background:
            raise NoContigPresentError
        
        mean = ma.mean(agg_cutcounts)
        variance = ma.var(agg_cutcounts, mean=mean, ddof=1)
        return mean, variance