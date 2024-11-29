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
        field_names,
        partition_cols,
        path,
        tmp_dir=None,
        *args,
        **kwargs
    ):
    assert len(field_names) == len(partition_cols)
    os.makedirs(path, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as temp_dir:
        tmp_prefix = '.'.join(field_names)
        temp_path = os.path.join(temp_dir, f'{tmp_prefix}.temp.parquet')
        to_parquet_high_compression(
            data_df,
            temp_path,
            partition_cols=partition_cols,
            *args,
            **kwargs
        )
        src, dest = get_src_and_dest_partioned_parquet(temp_path, path, partition_cols, field_names)
        dest_dir = os.path.dirname(dest)
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(src, dest)


def get_src_and_dest_partioned_parquet(src, dest, partition_cols, field_names):
    names = [
        f'{partition_col}={name}' 
        for partition_col, name in zip(partition_cols, field_names)
    ]
    parquet_file = [x for x in os.listdir(os.path.join(src, *names)) if x.endswith('.parquet')]
    assert len(parquet_file) == 1
    parquet_file = parquet_file[0]
    return os.path.join(src, *names, parquet_file), os.path.join(dest, *names, parquet_file)


def read_partioned_parquet(filename, partition_cols, partition_vals, columns=None):
    assert len(partition_cols) == len(partition_vals)
    return pd.read_parquet(
            filename,
            filters=[
                (partition_col, '==', partition_val) 
                for partition_col, partition_val in zip(partition_cols, partition_vals)
            ],
            engine='pyarrow',
            columns=columns
        )