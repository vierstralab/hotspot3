from functools import reduce
from scipy.signal import convolve, find_peaks
import numpy as np
import numpy.ma as ma
import scipy.stats as st
from scipy.stats import false_discovery_control
import pywt
import pandas as pd
import gc
from scipy.special import logsumexp


def negbin_neglog10pvalue(x, r, p):
    x = ma.asarray(x)
    r = ma.asarray(r)
    p = ma.asarray(p)
    assert r.shape == p.shape, "r and p should have the same shape"
    if len(r.shape) == 0:
        resulting_mask = x.mask
    else:
        resulting_mask = reduce(ma.mask_or, [x.mask, r.mask, p.mask])
        r = r[~resulting_mask]
        p = p[~resulting_mask]
    result = ma.masked_where(resulting_mask, np.zeros(x.shape, dtype=np.float32))
    result[~resulting_mask] = -st.nbinom.logsf(x[~resulting_mask] - 1, r, 1 - p) / np.log(10)
    return result.astype(np.float32)


def calc_neglog10fdr(neglog10_pvals, fdr_method='bh'):
    neglog10_fdr = np.empty(neglog10_pvals.shape, dtype=np.float16)
    not_nan = ~np.isnan(neglog10_pvals)
    neglog10_fdr[~not_nan] = np.nan
    neglog10_fdr[not_nan] = -logfdr_from_logpvals(
        -neglog10_pvals[not_nan] * np.log(10), method=fdr_method) / np.log(10)
    return neglog10_fdr


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
    # TODO: make work with log10 p-values
    log_pvals = np.asarray(log_pvals).astype(dtype=dtype) # no further input checking yet
    if log_pvals.ndim != 1:
        raise NotImplementedError("Only 1D arrays are supported.")

    m = log_pvals.shape[0]
    order = np.argsort(log_pvals) # ascending order, so that the lowest p-values are first
    
    log_pvals = log_pvals[order]

    log_i = np.log(np.arange(1, m + 1, dtype=dtype))
    log_pvals += np.log(m) - log_i  # p_adj = p * m / i => log10(p_adj) = log10(p) + log10(m) - log10(i)
    if method == 'by':
        log_pvals += logsumexp(-log_i)
    del log_i # optimization to free memory
    gc.collect()
    np.minimum.accumulate(log_pvals[::-1], out=log_pvals[::-1])
    log_pvals_copy = log_pvals.copy() # since array is float32, copying is more memory effient than argsort int64
    log_pvals[order] = log_pvals_copy
    return np.clip(log_pvals, a_min=None, a_max=0)


def nan_moving_sum(masked_array, window, dtype=None, position_skip_mask=None) -> ma.MaskedArray:
    if not isinstance(masked_array, ma.MaskedArray):
        masked_array = ma.masked_invalid(masked_array)

    if dtype is None:
        dtype = masked_array.dtype
    else:
        if dtype != masked_array.dtype:
            masked_array = masked_array.astype(dtype)

    data = masked_array.filled(0)
    if position_skip_mask is not None:
        assert position_skip_mask.shape == data.shape, "position_skip_mask should have the same shape as data"
        data[position_skip_mask] = 0

    conv_arr = np.ones(window, dtype=dtype)
    result = convolve(data, conv_arr, mode='same')
    return ma.array(result, mask=masked_array.mask)


def hotspots_from_log10_fdr_vectorized(log10_fdr_array, fdr_threshold, min_width):
    """
    Merge adjacent base pairs in a NumPy array where log10(FDR) is below the threshold.

    Parameters:
        - fdr_path: Path to the partitioned parquet file(s) containing the log10(FDR) values.
        - threshold: FDR threshold for merging regions.
        - min_width: Minimum width for a region to be called a hotspot.

    Returns:
        - pd.DataFrame: DataFrame containing the hotspots in bed format.
    """
    below_threshold = log10_fdr_array >= -np.log10(fdr_threshold)
    # Diff returns -1 for transitions from True to False, 1 for transitions from False to True
    boundaries = np.diff(below_threshold.astype(np.int8), prepend=0, append=0).astype(np.int8)

    region_starts = np.where(boundaries == 1)[0]
    region_ends = np.where(boundaries == -1)[0]

    valid_widths = (region_ends - region_starts) >= min_width
    region_starts = region_starts[valid_widths]
    region_ends = region_ends[valid_widths]

    max_log10_fdrs = np.empty(region_ends.shape)
    for i in range(len(region_starts)):
        start = region_starts[i]
        end = region_ends[i]
        max_log10_fdrs[i] = np.max(log10_fdr_array[start:end])

    return pd.DataFrame({
        'start': region_starts,
        'end': region_ends,
        'max_neglog10_fdr': max_log10_fdrs
    })


def convolve1d_fast(arr, ker, mode='wrap', origin=0):
    pad_width = len(ker) // 2
    padded_arr = np.pad(arr, pad_width, mode=mode)
    conv_result = convolve(padded_arr, ker, mode='valid')
    return np.roll(conv_result[:len(arr)], origin)


def convolve_d(h_t, v_j_1, j):
    '''
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j)
    '''
    ker = np.zeros(len(h_t) * 2**(j - 1))
    for i, h in enumerate(h_t):
        ker[i * 2**(j - 1)] = h

    return convolve1d_fast(v_j_1, ker, mode='wrap', origin=-len(ker) // 2)


def convolve_s(g_t, v_j, j):
    '''
    (j-1)th level synthesis from w_j, w_j
    see function circular_convolve_d
    '''
    g_ker = np.zeros(len(g_t) * 2**(j - 1))

    for i, g in enumerate(g_t):
        g_ker[i * 2**(j - 1)] = g

    return convolve1d_fast(v_j,
                        np.flip(g_ker),
                        mode='wrap',
                        origin=(len(g_ker) - 1) // 2)


def modwt(x, filters, level):
    '''
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    '''
    wavelet = pywt.Wavelet(filters)
    g = wavelet.dec_lo
    g_t = np.array(g) / np.sqrt(2)
    v_j_1 = x.astype(np.float32)
    for j in range(level):
        v_j_1 = convolve_d(g_t, v_j_1, j + 1)
    return v_j_1


def imodwt(v_j, filters, level):
    ''' inverse modwt '''
    wavelet = pywt.Wavelet(filters)
    g = wavelet.dec_lo
    g_t = np.array(g) / np.sqrt(2)
    for jp in range(level):
        j = level - jp - 1
        v_j = convolve_s(g_t, v_j, j + 1)
    return v_j


def modwt_smooth(x, filters, level):
    w = modwt(x, filters, level) # last level approximation
    return imodwt(w, filters, level)


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


def filter_peaks_summits_within_hotspots(peaks_coordinates, hotspot_starts, hotspot_ends):
    """
    Filter peaks that are in hotspots.
    Returns:
        - valid_hotspot_starts: starts of hotspots that contain peaks
        - valid_hotspot_ends: ends of hotspots that contain peaks
        - valid_mask: mask of peaks that are in hotspots
    """
    summits = peaks_coordinates[:, 1]
    closest_left_index = np.searchsorted(hotspot_starts, summits, side='right') - 1
    valid_mask = (closest_left_index >= 0) & (summits < hotspot_ends[closest_left_index])

    valid_hotspot_starts = hotspot_starts[closest_left_index[valid_mask]]
    valid_hotspot_ends = hotspot_ends[closest_left_index[valid_mask]]
    
    return valid_hotspot_starts, valid_hotspot_ends, valid_mask


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


def find_varwidth_peaks(signal: np.ndarray, hotspot_starts, hotspot_ends, min_width=20):
    """
    Find variable width peaks within hotspots.

    Parameters:
        - signal: 1D array of smoothed signal values.
        - hotspot_starts: 1D array of hotspot start positions.
        - hotspot_ends: 1D array of hotspot end positions.
        - min_width: Minimum width of peaks.
    
    Returns:
        - peaks_in_hotspots_trimmed: 2D array of start, summit, and end positions for each peak.
        - threshold_heights: 1D array of threshold heights for each peak.
    """
    peaks_coordinates = find_closest_min_peaks(signal)
    hs_left, hs_right, in_hs_mask = filter_peaks_summits_within_hotspots(peaks_coordinates, hotspot_starts, hotspot_ends)
    peaks_in_hotspots = peaks_coordinates[in_hs_mask, :]

    peaks_in_hotspots_trimmed, threshold_heights = trim_at_threshold(signal, peaks_in_hotspots)
    peaks_in_hotspots_trimmed[:, 0] = np.maximum(peaks_in_hotspots_trimmed[:, 0], hs_left)
    peaks_in_hotspots_trimmed[:, 2] = np.minimum(peaks_in_hotspots_trimmed[:, 2], hs_right)
    
    width_mask = (peaks_in_hotspots_trimmed[:, 2] - peaks_in_hotspots_trimmed[:, 0]) >= min_width
    return peaks_in_hotspots_trimmed[width_mask], threshold_heights[width_mask]
