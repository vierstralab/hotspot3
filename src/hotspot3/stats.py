import numpy as np
import numpy.ma as ma
import scipy.stats as st
import gc
from scipy.special import logsumexp, gammaln, betainc, hyp2f1, betaln
import itertools
from sortedcontainers import SortedList


# Calculate p-values and FDR
def negbin_neglog10pvalue(x: np.ndarray, r: np.ndarray, p: np.ndarray) -> np.ndarray:
    assert r.shape == p.shape, "r and p should have the same shape"

    result = logpval_for_dtype(x, r, p, dtype=np.float32, calc_type="betainc").astype(np.float16)
    low_precision = np.isinf(result)
    assert not np.any(np.isnan(result)), "Some p-values are NaN for betainc method"
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
    return result


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

def logfdr_from_logpvals(log_pvals, *, method='bh', dtype=np.float32):
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

    m = log_pvals.shape[0]
    order = np.argsort(log_pvals) # can save argsort as uint32
    
    log_pvals = log_pvals[order]

    log_i = np.log(np.arange(1, m + 1, dtype=dtype))
    log_pvals += np.log(m).astype(dtype) - log_i  # p_adj = p * m / i => log10(p_adj) = log10(p) + log10(m) - log10(i)
    if method == 'by':
        log_pvals += logsumexp(-log_i)
    del log_i
    gc.collect()
    np.minimum.accumulate(log_pvals[::-1], out=log_pvals[::-1])
    log_pvals_copy = log_pvals.copy() # since array is float32, copying is more memory effient than argsort int64
    log_pvals[order] = log_pvals_copy
    return np.clip(log_pvals, a_min=None, a_max=0)


# Find peaks



def cast_to_original_shape(data, original_shape, original_mask, step):
    res = np.full(original_shape, np.nan, dtype=data.dtype)
    res[::step] = data
    return ma.masked_where(original_mask, res)


def calc_epsilon_and_epsilon_mu(r, p, tr, step=None): # Can rewrite for code clarity
    if len(r.shape) == 0:
        eps = betainc(tr, r, p)
        eps_mu = eps * ((tr * (1 - p) / p + 1) / r - 1)
        return eps, eps_mu
    r = ma.masked_invalid(r)
    p = ma.masked_invalid(p)
    original_shape = r.shape
    original_mask = r.mask

    if step is not None:
        r = r[::step]
        p = p[::step]

    valid_mask = ~r.mask & ~p.mask

    eps = ma.masked_array(np.zeros(r.shape), mask=~valid_mask)
    eps[valid_mask] = betainc(tr, r[valid_mask], p[valid_mask])

    eps_mu = np.ma.masked_array(np.zeros(r.shape), mask=~valid_mask)
    eps_mu[valid_mask] = eps[valid_mask] * ((tr * (1 - p[valid_mask]) / p[valid_mask] + 1) / r[valid_mask] - 1)

    return (
        cast_to_original_shape(eps, original_shape, original_mask, step),
        cast_to_original_shape(eps_mu, original_shape, original_mask, step)
    )



def fast_nb_pmf(k, r, pmf_coef, pmf_const):
    return np.exp(gammaln(k + r) - gammaln(k + 1) + k * pmf_coef + pmf_const)


def calc_pmf_coefs(r, p, tr):
    r = np.ma.masked_invalid(r)
    p = np.ma.masked_invalid(p)
    
    valid_mask = ~r.mask & ~p.mask
    
    pmf_coef = np.ma.masked_array(np.zeros(r.shape), mask=~valid_mask)
    pmf_const = np.ma.masked_array(np.zeros(r.shape), mask=~valid_mask)
    
    pmf_coef[valid_mask] = np.log(p[valid_mask])
    pmf_const[valid_mask] = (
        np.log(1 - p[valid_mask]) * r[valid_mask]
        - gammaln(r[valid_mask])
        - np.log1p(-betainc(tr, r[valid_mask], p[valid_mask]))
    )
    
    return pmf_coef, pmf_const

def fix_inf_pvals(neglog_pvals, fname):
    infs = np.isinf(neglog_pvals)
    n_infs = np.sum(infs) 
    if n_infs > 0:
        np.savetxt(fname, np.where(infs)[0], fmt='%d')
        neglog_pvals[infs] = 300
    return neglog_pvals



