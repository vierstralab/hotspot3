from scipy.signal import find_peaks
import numpy as np
import numpy.ma as ma
import scipy.stats as st
import gc
from scipy.special import logsumexp, gammaln, betainc, hyp2f1, betaln
from hotspot3.signal_smoothing import nan_moving_sum
import itertools


# Calculate p-values and FDR
def p_and_r_from_mean_and_var(mean: np.ndarray, var: np.ndarray):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = ma.asarray(mean ** 2 / (var - mean), dtype=np.float32)
        p = ma.asarray(1 - mean / var, dtype=np.float32)
    return p, r


def negbin_neglog10pvalue(x: np.ndarray, r: np.ndarray, p: np.ndarray) -> np.ndarray:
    assert r.shape == p.shape, "r and p should have the same shape"

    result = logpval_for_dtype(x, r, p, dtype=np.float32, calc_type="nbinom").astype(np.float16)
    low_precision = ~np.isfinite(result)
    for precision, method in itertools.product(
        (np.float32, np.float64),
        ("nbinom", "beta", "hyp2f")
    ):
        if precision == np.float32 and method == "nbinom":
            continue
        if np.any(low_precision):
            new_pvals = logpval_for_dtype(
                x[low_precision],
                r[low_precision],
                p[low_precision],
                dtype=precision,
                calc_type=method
            )
            result[low_precision] = new_pvals
            low_precision[low_precision] = ~np.isfinite(new_pvals)
        else:
            break

    n = low_precision.sum()
    if n > 0:
        print(n, "p-values are still inf or nan.")
    result /= -np.log(10).astype(result.dtype)
    return result


def logpval_for_dtype(x: np.ndarray, r: np.ndarray, p: np.ndarray, dtype=None, calc_type="nbinom") -> np.ndarray:
    """
    Implementation of log(pval) with high precision
    """
    mask = x > 0
    x = np.asarray(x, dtype=dtype)[mask]
    r = np.asarray(r, dtype=dtype)[mask]
    p = np.asarray(p, dtype=dtype)[mask]
    
    result = np.zeros(mask.shape, dtype=dtype)
    if calc_type == 'nbinom':
        p_vals = st.nbinom.logsf(x - 1, r, 1 - p)
    elif calc_type == 'beta':
        p_vals = logpval_for_dtype_betainc(x, r, p)
    elif calc_type == 'hyp2f':
        p_vals = logpval_for_dtype_hyp2f(x, r, p)
    else:
        raise ValueError(f"Unknown p-value calculation type: {calc_type}")
    result[mask] = p_vals
    return result


def logpval_for_dtype_betainc(x: np.ndarray, r: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.log(betainc(x, r, p, dtype=r.dtype))


def logpval_for_dtype_hyp2f(x: np.ndarray, r: np.ndarray, p: np.ndarray) -> np.ndarray:
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
def filter_peaks_summits_within_regions(peaks_coordinates: np.ndarray, starts, ends):
    """
    Filter peaks which summits are within the regions defined by starts and ends.

    Parameters:
        - peaks_coordinates: np.array of shape (n_peaks, 3) with peak starts, summits and ends
        - starts: 1D array of regions' starts
        - ends: 1D array of regions' ends

    Returns:
        - filtered_peaks_mask: 1D boolean array of peaks that are within the regions
    """
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    summits = peaks_coordinates[:, 1]

    closest_left_index = np.searchsorted(starts, summits, side='right') - 1
    filtered_peaks_mask = (closest_left_index >= 0) & (summits < ends[closest_left_index])
    
    return filtered_peaks_mask


def find_closest_min_peaks(signal):
    """
    Find peak summits (maxima) and peak starts and ends (closest local minima).
    Returns:
        - peaks_coordinates: np.array of shape (n_peaks, 3) with peak starts, summits and ends
    """
    maxima, _ = find_peaks(signal)
    minima, _ = find_peaks(-signal)
    total_dif = len(maxima) - (len(minima) - 1)
    if total_dif == 1:
        fist_minima_pos = minima[0]
        fist_maxima_pos = maxima[0]
        if fist_minima_pos < fist_maxima_pos:
            padding = (0, 1)
        else:
            padding = (1, 0)
    elif total_dif == 0:
        padding = 0
    else:
        padding = 1
    minima = np.pad(minima, padding, mode='constant', constant_values=(0, len(signal) - 1))
    peaks_coordinates = np.zeros([len(maxima), 3], dtype=int) # start, summit, end
    peaks_coordinates[:, 1] = maxima

    peaks_coordinates[:, 0] = minima[:len(maxima)]
    peaks_coordinates[:, 2] = minima[1:]
    return peaks_coordinates


def trim_at_threshold(signal, peaks_coordinates):
    """
    Trim peaks at the threshold height.
    Returns:
        - peaks_in_hotspots_trimmed: trimmed peaks coordinates
        - threshold_heights: heights of the threshold
    """
    heights = signal[peaks_coordinates]
    heights[:, 1] = heights[:, 1] / 2
    threshold_heights = np.max(heights, axis=1)

    peaks_in_hotspots_trimmed = np.zeros(peaks_coordinates.shape, dtype=np.int32)
    for i, (left, summit, right) in enumerate(peaks_coordinates):
        threshold = threshold_heights[i]
        new_left = summit - np.argmax(signal[left:summit + 1][::-1] <= threshold)

        new_right = summit + np.argmax(signal[summit:right + 1] <= threshold) - 1
        peaks_in_hotspots_trimmed[i, :] = np.array([new_left, summit, new_right])
    return peaks_in_hotspots_trimmed, threshold_heights


def find_varwidth_peaks(signal: np.ndarray, starts=None, ends=None):
    """
    Find variable width peaks within hotspots.

    Parameters:
        - signal: 1D array of smoothed signal values.
        - starts: 1D array of start positions of significant stretches.
        - ends: 1D array of end positions of significant stretches.
    
    Returns:
        - peaks_in_hotspots_trimmed: 2D array of start, summit, and end positions for each peak.
        - threshold_heights: 1D array of threshold heights for each peak.
    """
    if starts is None:
        assert ends is None
        starts = np.zeros(1)
        ends = np.zeros(1)
    peaks_coordinates = find_closest_min_peaks(signal)
    in_sign_stretch = filter_peaks_summits_within_regions(peaks_coordinates, starts, ends)
    
    peaks_in_regions = peaks_coordinates[in_sign_stretch, :]

    peaks_in_hotspots_trimmed, threshold_heights = trim_at_threshold(signal, peaks_in_regions)
    
    return peaks_in_hotspots_trimmed, threshold_heights


def calc_rmsea(obs, unique_cutcounts, r, p, tr):
    N = sum(obs)
    exp = st.nbinom.pmf(unique_cutcounts, r, 1 - p) / st.nbinom.cdf(tr - 1, r, 1 - p) * N
    # chisq = sum((obs - exp) ** 2 / exp)
    G_sq = 2 * sum(obs * np.log(obs / exp))
    df = len(obs) - 2
    return np.sqrt((G_sq / df - 1) / (N - 1))


def cast_to_original_shape(data, original_shape, original_mask, step):
    res = np.full(original_shape, np.nan, dtype=data.dtype)
    res[::step] = data
    return ma.masked_where(original_mask, res)


def calc_epsilon_and_epsilon_mu(r, p, tr, step=None): # Can rewrite for code clarity
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

    if len(original_shape) == 0:
        return eps.compressed()[0], eps_mu.compressed()[0]
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


def calc_g_sq_for_k(
        k, r,
        pmf_coef, pmf_const,
        agg_cov_masked,
        bg_sum_mappable,
        bg_window,
        position_skip_mask,
        dtype,
        step,
):
    # same as: exp = st.nbinom.pmf(k, r, 1 - p) / st.nbinom.cdf(tr - 1, r, 1 - p) * bg_sum_mappable
    exp = fast_nb_pmf(k, r, pmf_coef, pmf_const) * bg_sum_mappable

    obs = nan_moving_sum(
        agg_cov_masked == k,
        bg_window,
        position_skip_mask=position_skip_mask,
        dtype=dtype,
    ) * step
    ratio = obs / exp
    ratio[ratio == 0] = 1
    return obs * np.ma.log(ratio) * 2


def calc_rmsea_all_windows(
        r, p, tr,
        bg_sum_mappable,
        agg_cov_masked,
        bg_window,
        position_skip_mask,
        dtype,
        step=20,
    ):
    """
    Calculate RMSEA for all sliding windows with given stride.
    Stride is required to be a divisor of the window size.
    """
    assert (bg_window - 1) % step == 0, "Stride should be a divisor of the window size."
    initial_shape = r.shape
    initial_mask = r.mask
    agg_cov_masked = agg_cov_masked[::step]
    position_skip_mask = position_skip_mask[::step]
    bg_sum_mappable = bg_sum_mappable[::step]
    r = r[::step]
    p = p[::step]
    bg_window = (bg_window - 1) // step + 1

    pmf_coef, pmf_const = calc_pmf_coefs(r, p, tr)
    G_sq = np.ma.masked_where(r.mask, np.zeros_like(r))
    for k in range(tr):
        G_sq += calc_g_sq_for_k(
            k, r,
            pmf_coef, pmf_const,
            agg_cov_masked,
            bg_sum_mappable,
            bg_window,
            position_skip_mask,
            dtype,
            step,
        )

    G_sq = np.ma.masked_where(~np.isfinite(G_sq), G_sq)
    rmsea_stride = np.sqrt(np.clip(G_sq / (tr - 2) - 1, 0, None) / (bg_sum_mappable - 1))

    return cast_to_original_shape(rmsea_stride, initial_shape, initial_mask, step)
