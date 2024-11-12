import pandas as pd
import numpy as np
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


def parallel_func_error_handler(func):
    @functools.wraps(func)
    def wrapper(processor: 'ChromosomeProcessor', *args):
        try:
            return func(processor, *args)
        except:
            processor.logger.exception(f"Exception occured in {func.__name__} for chromosome {processor.chrom_name}")
            raise
    return wrapper


def is_iterable(obj):
    if isinstance(obj, pd.DataFrame) or isinstance(obj, str):
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def wrap_masked(func) -> ma.MaskedArray:
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
            return wrap_if_1d(mask, result)
        elif isinstance(result, tuple):
            return tuple(wrap_if_1d(mask, r) for r in result)
        else:
            return result
    return wrapper

def wrap_if_1d(mask: np.ndarray, result: np.ndarray):
    if result.ndim == 0:
        return result
    return ma.masked_where(mask, result)

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
        array = np.asarray(array, dtype=np.float32)
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
