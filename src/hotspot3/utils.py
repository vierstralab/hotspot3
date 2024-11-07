import pandas as pd
import os
import numpy as np
import shutil
import pyBigWig
import functools
import numpy.ma as ma
from hotspot3.models import NotEnoughDataForContig



def ensure_contig_exists(func):
    """
    Decorator for functions that require a contig to be present in the input data.

    Returns None if the contig is not present.
    Otherwise, returns the result of the function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotEnoughDataForContig:
            return None
    return wrapper


def is_iterable(obj):
    if isinstance(obj, pd.DataFrame) or isinstance(obj, str):
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False
    

def normalize_density(density, total_cutcounts):
    return (density / total_cutcounts * 1_000_000).astype(np.float32)


# I/O functions
def read_chrom_sizes(chrom_sizes):
    if chrom_sizes is None:
        raise NotImplementedError("hg38 chromosome sizes are not embedded yet. Please provide a chromosome sizes file.")
    return pd.read_table(
        chrom_sizes,
        header=None,
        names=['chrom', 'size']
    ).set_index('chrom')['size'].to_dict()


def to_parquet_high_compression(df: pd.DataFrame, outpath, **kwargs):
    df.to_parquet(
        outpath,
        engine='pyarrow',
        compression='zstd',
        use_dictionary=True,
        index=False,
        partition_cols=['chrom'],
        **kwargs
    )


def df_to_bigwig(df: pd.DataFrame, outpath, chrom_sizes: dict, col='value'):
    with pyBigWig.open(outpath, 'w') as bw:
        bw.addHeader(list(chrom_sizes.items()))
        chroms = df['chrom'].to_list()
        starts = df['start'].to_list()
        ends = df['end'].to_list()
        values = df[col].to_list()
        bw.addEntries(chroms, starts, ends=ends, values=values)


def delete_path(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def wrap_masked(func, sampling_step=1) -> ma.MaskedArray:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        masked_arrays = [arg for arg in args if isinstance(arg, ma.MaskedArray)]
        masked_arrays += [value for value in kwargs.values() if isinstance(value, ma.MaskedArray)]

        if len(masked_arrays) > 0: # introduces overhead, maybe ignore this check
            assert all(masked_arrays[0].shape == ma.shape for ma in masked_arrays), 'All masked arrays should have the same shape'
            assert all(np.all(masked_arrays[0].mask == ma.mask) for ma in masked_arrays), 'All masked arrays should have the same mask'
            mask = masked_arrays[0].mask
        else:
            return func(*args, **kwargs)
        args = [compress_masked_arg(arg) for arg in args]
        kwargs = {key: compress_masked_arg(value) for key, value in kwargs.items()}
        result = func(*args, **kwargs)

        if isinstance(result, np.ndarray):
            return ma.masked_where(mask, result)
        elif isinstance(result, tuple):
            return tuple(ma.masked_where(mask, r) for r in result)
        else:
            return result
    return wrapper


def compress_masked_arg(arg):
    if isinstance(arg, ma.MaskedArray):
        return arg.filled(np.nan)
    else:
        return arg

def correct_offset(func):
    """
    Correct offset of a trailing running window to make it centered.
    """
    @functools.wraps(func)
    def wrapper(self, array, window, *args, **kwargs):
        assert window % 2 == 1, "Window size should be odd"
        if len(array) < window:
            window = len(array)
        offset = window // 2
        result = func(
            self,
            np.pad(array, (0, offset), mode='constant', constant_values=np.nan),
            window,
            *args,
            **kwargs
        )
        return result[offset:]
    return wrapper

@wrap_masked
def interpolate_nan(array):
    subsampled_indices = np.where(~np.isnan(array))[0]
    subsampled_arr = array[subsampled_indices]
    try:
        return np.interp(
            np.arange(array.shape[0], dtype=np.uint32),
            subsampled_indices,
            subsampled_arr,
            left=None,
            right=None,
        )
    except ValueError:
        return np.full(array.shape, np.nan)


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
