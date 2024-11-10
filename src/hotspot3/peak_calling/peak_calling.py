import numpy as np
from scipy.signal import find_peaks


def find_stretches(arr: np.ndarray):
    """
    Find stretches of True values in a boolean array.

    Parameters:
        - arr: np.ndarray 1D boolean array.

    Returns:
        - start: np.ndarray of start positions of stretches
        - end: end positions of stretches
    """
    boundaries = np.diff(arr.astype(np.int8), prepend=0, append=0)
    start = np.where(boundaries == 1)[0]
    end = np.where(boundaries == -1)[0]
    return start, end


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

        new_right = summit + np.argmax(signal[summit:right + 1] <= threshold)
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

