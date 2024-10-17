from functools import reduce
from scipy.signal import find_peaks
import numpy as np
import numpy.ma as ma
import scipy.stats as st
import pandas as pd
import gc
from scipy.special import logsumexp


# Calculate p-values and FDR
def p_and_r_from_mean_and_var(mean, var):
    r = mean ** 2 / (var - mean) # mean
    p = 1 - mean / var # (var - mean) / var
    return p, r


def negbin_neglog10pvalue(x: ma.MaskedArray, r, p) -> np.ndarray:
    x = ma.asarray(x)
    r = ma.asarray(r)
    p = ma.asarray(p)
    assert r.shape == p.shape, "r and p should have the same shape"
    resulting_mask = x.mask.copy()
    # in masked arrays, mask is True for masked values
    if len(r.shape) != 0:
        resulting_mask = reduce(ma.mask_or, [resulting_mask, r.mask, p.mask])
        r = r[~resulting_mask].compressed()
        p = p[~resulting_mask].compressed()
    x = x[~resulting_mask].compressed()

    result = np.empty(resulting_mask.shape, dtype=np.float16)
    result[resulting_mask] = np.nan
    result[~resulting_mask] = -st.nbinom.logsf(x - 1, r, 1 - p) / np.log(10)
    return result


def calc_neglog10fdr(neglog10_pvals, fdr_method='bh'):
    neglog10_fdr = np.empty(neglog10_pvals.shape, dtype=np.float16)
    not_nan = ~np.isnan(neglog10_pvals)
    neglog10_fdr[~not_nan] = np.nan
    neglog10_fdr[not_nan] = -logfdr_from_logpvals(
        -neglog10_pvals[not_nan] * np.log(10), method=fdr_method
    ) / np.log(10)
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
    log_pvals = np.asarray(log_pvals).astype(dtype=dtype) # no further input checking yet
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


# Find hotspots
def find_stretches(arr: np.ndarray):
    """
    Find stretches of True values in a boolean array.
    Returns:
        - start: start positions of stretches
        - end: end positions of stretches
    """
    boundaries = np.diff(arr.astype(np.int8), prepend=0, append=0)
    start = np.where(boundaries == 1)[0]
    end = np.where(boundaries == -1)[0]
    return start, end


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

    region_starts, region_ends = find_stretches(below_threshold)
    
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


# Find peaks
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
    _, _, in_hs_mask = filter_peaks_summits_within_hotspots(
        peaks_coordinates,
        hotspot_starts,
        hotspot_ends,
    )
    peaks_in_hotspots = peaks_coordinates[in_hs_mask, :]

    peaks_in_hotspots_trimmed, threshold_heights = trim_at_threshold(signal, peaks_in_hotspots)
    
    width_mask = (peaks_in_hotspots_trimmed[:, 2] - peaks_in_hotspots_trimmed[:, 0]) >= min_width
    return peaks_in_hotspots_trimmed[width_mask], threshold_heights[width_mask]