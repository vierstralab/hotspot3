import pandas as pd
import os
import shutil
import pyBigWig


def read_chrom_sizes(chrom_sizes):
    if chrom_sizes is None:
        raise NotImplementedError("hg38 chromosome sizes are not embedded yet. Please provide a chromosome sizes file.")
    return pd.read_table(
        chrom_sizes,
        header=None,
        names=['chrom', 'size']
    ).set_index('chrom')['size'].to_dict()


def to_parquet_high_compression(df: pd.DataFrame, outpath, compression_level=22, **kwargs):
    df.to_parquet(
        outpath,
        engine='pyarrow',
        compression='zstd',
        use_dictionary=True,
        index=False,
        partition_cols=['chrom'],
        compression_level=compression_level,
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