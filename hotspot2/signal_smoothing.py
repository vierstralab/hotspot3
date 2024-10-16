import numpy as np
import numpy.ma as ma
from scipy.signal import convolve
import scipy.stats as st
import pywt


def convolve1d_fast(arr, ker, mode='wrap', origin=0):
    pad_width = len(ker) // 2
    padded_arr = np.pad(arr, pad_width, mode=mode)
    conv_result = convolve(padded_arr, ker, mode='valid')
    return np.roll(conv_result[:len(arr)], origin)


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


def imodwt(v_j, filters, level):
    ''' inverse modwt '''
    wavelet = pywt.Wavelet(filters)
    g = wavelet.dec_lo
    g_t = np.array(g) / np.sqrt(2)
    for jp in range(level):
        j = level - jp - 1
        v_j = convolve_s(g_t, v_j, j + 1)
    return v_j


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


def modwt_smooth(x, filters, level):
    w = modwt(x, filters, level) # last level approximation
    return imodwt(w, filters, level)


def nan_moving_sum(masked_array, window, dtype=None, position_skip_mask: np.ndarray=None) -> ma.MaskedArray:
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


def calc_rmsea(obs, unique_cutcounts, r, p, tr):
    N = sum(obs)
    exp = st.nbinom.pmf(unique_cutcounts, r, 1 - p) / st.nbinom.cdf(tr - 1, r, 1 - p) * N
    # chisq = sum((obs - exp) ** 2 / exp)
    G_sq = 2 * sum(obs * np.log(obs / exp))
    df = len(obs) - 2
    return np.sqrt((G_sq / df - 1) / (N - 1))


def calc_epsilon(r, p, tr):
    return st.nbinom(r, 1 - p).pmf(tr) / (1 - p)
