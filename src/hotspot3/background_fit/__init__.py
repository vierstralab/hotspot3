import numpy as np
from typing import Union
from hotspot3.models import WindowedFitResults, FitResults


def rolling_view_with_nan_padding(arr, points_in_window=501, interpolation_step=1000):
    """
    Creates a 2D array where each row is a shifted view of the original array with NaN padding.
    Only every 'subsample_step' row is taken to reduce memory usage.

    Parameters:
        arr (np.ndarray): Input array of shape (n,)
        window_size (int): The total size of the shift (default is 501 for shifts from -250 to +250)
        subsample_step (int): Step size for subsampling (default is 1000)

    Returns:
        np.ndarray: A subsampled array with NaN padding, of shape (n // subsample_step, window_size)
    """
    n = arr.shape[0]
    assert points_in_window % 2 == 1, "Window size must be odd to have a center shift of 0."
    
    pad_width = (points_in_window - 1) // 2
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
    subsample_count = (n + interpolation_step - 1) // interpolation_step
    shape = (subsample_count, points_in_window)
    strides = (padded_arr.strides[0] * interpolation_step, padded_arr.strides[0])
    subsampled_view = np.lib.stride_tricks.as_strided(padded_arr, shape=shape, strides=strides)
 
    return subsampled_view.T


def calc_g_sq(obs, exp):
    valid = (exp != 0) & (obs != 0)
    with np.errstate(over='ignore'):
        ratio = np.divide(obs, exp, out=np.zeros_like(obs), where=valid)
    ratio = np.where(valid, ratio, 1)
    return obs * np.log(ratio) * 2

def rolling_view_with_nan_padding(arr, points_in_window=501, interpolation_step=1000):
    """
    Creates a 2D array where each row is a shifted view of the original array with NaN padding.
    Only every 'subsample_step' row is taken to reduce memory usage.

    Parameters:
        arr (np.ndarray): Input array of shape (n,)
        window_size (int): The total size of the shift (default is 501 for shifts from -250 to +250)
        subsample_step (int): Step size for subsampling (default is 1000)

    Returns:
        np.ndarray: A subsampled array with NaN padding, of shape (n // subsample_step, window_size)
    """
    n = arr.shape[0]
    assert points_in_window % 2 == 1, "Window size must be odd to have a center shift of 0."
    
    pad_width = (points_in_window - 1) // 2
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
    subsample_count = (n + interpolation_step - 1) // interpolation_step
    shape = (subsample_count, points_in_window)
    strides = (padded_arr.strides[0] * interpolation_step, padded_arr.strides[0])
    subsampled_view = np.lib.stride_tricks.as_strided(padded_arr, shape=shape, strides=strides)
 
    return subsampled_view.T

def calc_chisq(obs, exp):
    with np.errstate(over='ignore'):
        return np.where((exp != 0) & (obs != 0), (obs - exp) ** 2 / exp, 0)


def calc_rmsea(obs, exp, N, df, min_df=1, stat='G_sq', where=None):
    assert stat in ('G_sq', 'chi_sq'), "Only G_sq and chi_sq statistics are supported"
    if stat == 'G_sq':
        G_sq = calc_g_sq(obs, exp)
    else:
        G_sq = calc_chisq(obs, exp)

    G_sq = np.sum(G_sq, axis=0, where=where)
    G_sq = np.divide(G_sq, df, out=np.zeros_like(G_sq), where=df>=min_df)
    rmsea = np.sqrt(np.maximum(G_sq - 1, 0) / (N - 1))
    rmsea = np.where(df >= min_df, rmsea, np.inf)
    return rmsea


def check_valid_fit(fit: Union[WindowedFitResults, FitResults]):
    return (fit.r > 0.) & (fit.p > 0.) & (fit.p < 1.) 
