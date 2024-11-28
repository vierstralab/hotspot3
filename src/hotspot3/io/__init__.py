import pandas as pd
import os
import shutil
import tempfile


def to_parquet_high_compression(df: pd.DataFrame, outpath, compression_level=22, partition_cols=None, **kwargs):
    if partition_cols is None:
        partition_cols = ['chrom']

    df.to_parquet(
        outpath,
        engine='pyarrow',
        compression='zstd',
        use_dictionary=True,
        index=False,
        partition_cols=partition_cols,
        compression_level=compression_level,
        **kwargs
    )


def parallel_write_partitioned_parquet(
        data_df,
        name,
        partition_col,
        path,
        tmp_dir=None,
        *args,
        **kwargs
    ):
    os.makedirs(path, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as temp_dir:
        temp_path = os.path.join(temp_dir, f'{name}.temp.parquet')
        to_parquet_high_compression(
            data_df,
            temp_path,
            partition_cols=[partition_col],
            *args,
            **kwargs
        )
        res_path = os.path.join(path, f'{partition_col}={name}')
        if os.path.exists(res_path):
            shutil.rmtree(res_path)
        shutil.move(os.path.join(temp_path, f'{partition_col}={name}'), path)


def read_partioned_parquet(filename, partition_col, partition_val, columns=None):
    return pd.read_parquet(
            filename,
            filters=[(partition_col, '==', partition_val)],
            engine='pyarrow',
            columns=columns
        )