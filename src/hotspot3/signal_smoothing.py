import numpy as np
import numpy.ma as ma
from scipy.signal import convolve
import pywt
import bottleneck as bn


def convolve1d_fast(arr, ker, mode='wrap', origin=0):
    pad_width = len(ker) // 2
    padded_arr = np.pad(arr, pad_width, mode=mode)
    conv_result = convolve(padded_arr, ker, mode='valid')
    return np.roll(conv_result[:len(arr)], origin)


def convolve_s(g_t, v_j, j):
    """
    (j-1)th level synthesis from w_j
    """
    dtype = v_j.dtype
    g_ker = np.zeros(len(g_t) * 2**(j - 1), dtype=dtype)

    for i, g in enumerate(g_t):
        g_ker[i * 2**(j - 1)] = g

    return convolve1d_fast(
        v_j,
        np.flip(g_ker),
        mode='wrap',
        origin=(len(g_ker) - 1) // 2
    )


def init_g_t(filters):
    wavelet = pywt.Wavelet(filters)
    g = wavelet.dec_lo
    return np.array(g, dtype=np.float32) / np.sqrt(2)


def imodwt(v_j, filters, level):
    ''' inverse modwt '''
    g_t = init_g_t(filters)
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
    dtype = v_j_1.dtype
    ker = np.zeros(len(h_t) * 2**(j - 1), dtype=dtype)
    for i, h in enumerate(h_t):
        ker[i * 2**(j - 1)] = h

    return convolve1d_fast(v_j_1, ker, mode='wrap', origin=-len(ker) // 2)


def modwt(x, filters, level):
    '''
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    '''
    g_t = init_g_t(filters)
    v_j_1 = x
    for j in range(level):
        v_j_1 = convolve_d(g_t, v_j_1, j + 1)
    return v_j_1


def modwt_smooth(x, filters, level):
    w = modwt(x, filters, level) # last level approximation
    return imodwt(w, filters, level)
