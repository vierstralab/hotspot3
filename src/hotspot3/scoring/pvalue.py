import numpy as np
import itertools

from hotspot3.io.logging import WithLogger
from hotspot3.helpers.models import WindowedFitResults
from hotspot3.scoring import logpval_for_dtype, fix_inf_pvals


class PvalueEstimator(WithLogger):

    def estimate_pvalues(self, agg_cutcounts: np.ndarray, fit_results: WindowedFitResults) -> np.ndarray:
        result = np.full_like(agg_cutcounts, np.nan, dtype=np.float16)
        r = fit_results.r
        p = fit_results.p
        mask = fit_results.enough_bg_mask
        data, invalid = self.negbin_neglog10pvalue(agg_cutcounts[mask], r[mask], p[mask])
        result[mask] = data
        if invalid is not None:
            ids = np.arange(result.shape[0])[mask][invalid]
            self.logger.critical(f"{self.name}: {len(ids)} p-values are NaN for betainc method, {ids}. Parameters: r={r[ids]}, p={p[ids]}, count={agg_cutcounts[ids]}")
            raise ValueError(f"{self.name}: {len(ids)} p-values are NaN for betainc method, {ids}")
        return result

    def negbin_neglog10pvalue(self, x: np.ndarray, r: np.ndarray, p: np.ndarray) -> np.ndarray:
        result = logpval_for_dtype(x, r, p, dtype=np.float32, calc_type="betainc").astype(np.float16)
        low_precision = np.isinf(result)
        invalid_pvals = np.isnan(result)
        if np.any(invalid_pvals):
            return result, invalid_pvals
        for precision, method in itertools.product(
            (np.float32, np.float64),
            ("betainc", "hyp2f1", "nbinom")
        ):
            if precision == np.float32 and method == "betainc":
                continue
            if np.any(low_precision):
                new_pvals = logpval_for_dtype(
                    x[low_precision],
                    r[low_precision],
                    p[low_precision],
                    dtype=precision,
                    calc_type=method
                )
                corrected_infs = np.isfinite(new_pvals)
                result[low_precision] = np.where(corrected_infs, new_pvals, result[low_precision])
                low_precision[low_precision] = ~corrected_infs

            else:
                break
        result /= -np.log(10).astype(result.dtype)
        return result, None

    def fix_inf_pvals(self, pvals):
        return fix_inf_pvals(pvals)