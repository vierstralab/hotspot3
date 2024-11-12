import pandas as pd
import numpy as np


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


def convert_to_score(array, mult, max_score=1000):
    return np.round(array * mult).astype(np.int64).clip(0, max_score)
