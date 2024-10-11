from functools import reduce
from scipy.signal import convolve
import numpy as np
import numpy.ma as ma
import scipy.stats as st
from statsmodels.stats.multitest import multipletests
from utils import read_df_for_chrom
import pandas as pd


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


def calc_log10fdr(log10_pvals, fdr_method='fdr_bh'):
    log_fdr = np.empty(log10_pvals.shape, dtype=np.float16)
    not_nan = ~np.isnan(log10_pvals)
    log_fdr[~not_nan] = np.nan
    log_fdr[not_nan] = -np.log10(multipletests(np.power(10, -log10_pvals[not_nan].astype(np.float64)), method=fdr_method)[1])
    return log_fdr.astype(np.float16)


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

    min_log10_fdr_values = np.empty(region_ends.shape)
    for i in range(len(region_starts)):
        start = region_starts[i]
        end = region_ends[i]
        min_log10_fdr_values[i] = np.max(log10_fdr_array[start:end])

    return pd.DataFrame({
        'start': region_starts,
        'end': region_ends,
        'max_neglog10_fdr': min_log10_fdr_values
    })