import pandas as pd
import os
import numpy as np
import shutil
import pyBigWig
import functools

from hotspot3.models import NoContigPresentError


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
        except NoContigPresentError:
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
        compression_level=22,
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
