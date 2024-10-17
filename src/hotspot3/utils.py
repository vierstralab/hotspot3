import pandas as pd
import os
import numpy as np
import shutil
import logging
import sys


def is_iterable(obj):
    if isinstance(obj, pd.DataFrame) or isinstance(obj, str):
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False
    

def set_logger_config(logger: logging.Logger, level: int):
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    

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


def delete_path(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
