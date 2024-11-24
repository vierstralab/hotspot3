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
