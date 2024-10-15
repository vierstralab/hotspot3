import pandas as pd
import dataclasses
import pysam
import io
import functools
import subprocess
import os
import numpy as np
import shutil


class NoContigPresentError(Exception):
    ...


@dataclasses.dataclass
class ProcessorOutputData:
    """
    Dataclass for storing the output of ChromosomeProcessor and GenomeProcessor methods.
    """
    identificator: str
    data_df: pd.DataFrame


def is_iterable(obj):
    if isinstance(obj, pd.DataFrame) or isinstance(obj, str):
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False
    

def read_chrom_sizes(chrom_sizes):
    if chrom_sizes is None:
        raise NotImplementedError("hg38 chromosome sizes are not embedded yet. Please provide a chromosome sizes file.")
    return pd.read_table(
        chrom_sizes,
        header=None,
        names=['chrom', 'size']
    ).set_index('chrom')['size'].to_dict()


def read_parquet_for_chrom(df_path, chrom_name, columns=None):
    return pd.read_parquet(
        df_path,
        filters=[('chrom', '==', chrom_name)],
        engine='pyarrow',
        columns=columns
    )


def df_to_tabix(df: pd.DataFrame, tabix_path):
    """
    Convert a DataFrame to a tabix-indexed file.
    Renames 'chrom' column to '#chr' if exists.

    Parameters:
        - df: DataFrame to convert - bed format. First columns (chr start end).
        - tabix_path: Path to the tabix-indexed file.

    Returns:
        - None
    """
    with pysam.BGZFile(tabix_path, 'w') as bgzip_out:
        with io.TextIOWrapper(bgzip_out, encoding='utf-8') as text_out:
            df.rename(columns={'chrom': '#chr'}).to_csv(text_out, sep='\t', index=False)

    pysam.tabix_index(tabix_path, preset='bed', force=True)


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

def normalize_density(density, total_cutcounts):
    return (density / total_cutcounts * 1_000_000).astype(np.float32)


def run_bam2_bed(bam_path, tabix_bed_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(script_dir, 'extract_cutcounts.sh')
    subprocess.run(['bash', script, bam_path, tabix_bed_path], check=True)


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
    