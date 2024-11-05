import numpy as np
import numpy.ma as ma
import scipy.stats as st
import gc
from scipy.special import logsumexp, betainc, hyp2f1, betaln
from hotspot3.models import WindowedFitResults


# Calculate p-values and FDR
def logpval_for_dtype(x: np.ndarray, r: np.ndarray, p: np.ndarray, dtype=None, calc_type="betainc") -> np.ndarray:
    """
    Implementation of log(pval) with high precision
    """
    mask = x > 0
    x = np.round(np.asarray(x, dtype=dtype)[mask])
    r = np.asarray(r, dtype=dtype)[mask]
    p = np.asarray(p, dtype=dtype)[mask]
    
    result = np.zeros(mask.shape, dtype=dtype)
    if calc_type == 'nbinom':
        p_vals = st.nbinom.logsf(x - 1, r, 1 - p)
    elif calc_type == 'betainc':
        p_vals = logpval_for_dtype_betainc(x, r, p)
    elif calc_type == 'hyp2f1':
        p_vals = logpval_for_dtype_hyp2f(x, r, p)
    else:
        raise ValueError(f"Unknown p-value calculation type: {calc_type}")
    result[mask] = p_vals
    return result


def logpval_for_dtype_betainc(x: np.ndarray, r: np.ndarray, p: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(betainc(x, r, p, dtype=r.dtype))


def logpval_for_dtype_hyp2f(x: np.ndarray, r: np.ndarray, p: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return (
            x * np.log(p) 
            + r * np.log(1 - p) 
            + np.log(hyp2f1(x + r, 1, x + 1, p, dtype=r.dtype))
            - np.log(x)
            - betaln(x, r, dtype=r.dtype)
        )

def logfdr_from_logpvals(log_pvals, *, method='bh', dtype=np.float32, m=None):
    """
    Reimplementation of scipy.stats.false_discovery_control to work with neglog-transformed p-values.
    Accepts log-transformed p-values and returns log-transformed FDR values.
    NOTE: Not log10-transformed and not negated!

    Parameters:
        - log_pvals: 1D array of log-transformed p-values. Most of them will be negative.
        - fdr_method: Method to use for FDR calculation. 'bh' = Benjamini-Hochberg, 'by' = Benjamini-Yekutieli.
    
    Returns:
        - log_fdr: 1D array of log-transformed FDR values. 
    """
    assert method in ['bh', 'by'], "Only 'bh' and 'by' methods are supported."
    log_pvals = np.asarray(log_pvals, dtype=dtype) # no further input checking yet
    if log_pvals.ndim != 1:
        raise NotImplementedError("Only 1D arrays are supported.")

    if m is None:
        m = log_pvals.shape[0]
    order = np.argsort(log_pvals) # can save argsort as uint32
    
    log_pvals = log_pvals[order]

    log_i = np.log(np.arange(1, log_pvals.shape[0] + 1, dtype=dtype))
    log_pvals += np.log(m).astype(dtype) - log_i  # p_adj = p * m / i => log10(p_adj) = log10(p) + log10(m) - log10(i)
    if method == 'by':
        log_pvals += logsumexp(-log_i)
    del log_i
    gc.collect()
    np.minimum.accumulate(log_pvals[::-1], out=log_pvals[::-1])
    log_pvals_copy = log_pvals.copy() # since array is float32, copying is more memory effient than argsort int64
    log_pvals[order] = log_pvals_copy
    return np.clip(log_pvals, a_min=None, a_max=0)


def fix_inf_pvals(neglog_pvals, fname): # TODO move somewhere else
    infs = np.isinf(neglog_pvals)
    n_infs = np.sum(infs) 
    if n_infs > 0:
        np.savetxt(fname, np.where(infs)[0], fmt='%d')
        neglog_pvals[infs] = 300
    return neglog_pvals


def calc_g_sq(obs, exp):
    valid = (exp != 0) & (obs != 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(obs, exp, out=np.zeros_like(obs), where=valid)
    ratio = np.where(valid, ratio, 1)
    return obs * np.log(ratio) * 2


def calc_chisq(obs, exp):
    return np.where((exp != 0) & (obs != 0), (obs - exp) ** 2 / exp, 0)


def calc_rmsea(G_sq, N, df, min_df=7):
    G_sq = np.divide(G_sq, df, out=np.zeros_like(G_sq), where=df >= min_df)
    rmsea = np.sqrt(np.maximum(G_sq - 1, 0) / (N - 1))
    rmsea = np.where(df >= min_df, rmsea, np.inf)
    return rmsea


def check_valid_fit(fit: WindowedFitResults):
    return (fit.r >= 0.) & (fit.p >= 0.) & (fit.p <= 1.) & fit.enough_bg_mask
