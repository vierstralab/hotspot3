import numpy as np
import scipy.stats as st
from hotspot3.stats import negbin_neglog10pvalue


class PvalueEstimator:
    def __init__(self, config):
        self.config = config

    def estimate_pvalues(self, 
            agg_cutcounts: np.ndarray, 
            r: np.ndarray, 
            p: np.ndarray,
            mask: np.ndarray,
            bad_fits=None
        ):
        result = np.full_like(mask, np.nan, dtype=np.float16)
        
        if bad_fits is not None:
            indx = bad_fits[:, 0].astype(int)
            param = bad_fits[:, 1]
            result[indx] = st.poisson.logsf(agg_cutcounts[indx] - 1, param)
            mask[indx] = False
        
        result[mask] = negbin_neglog10pvalue(agg_cutcounts[mask], r[mask], p[mask])
        return result

        