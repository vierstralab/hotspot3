import pandas as pd
import subprocess
import importlib.resources as pkg_resources
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

def run_bam2_bed(bam_path, tabix_bed_path, chromosomes=None):
    """
    Run bam2bed conversion script.
    """
    with pkg_resources.path('hotspot3.scripts', 'extract_cutcounts.sh') as script:
        chroms = ','.join(chromosomes) if chromosomes else ""
        subprocess.run(
            ['bash', script, bam_path, tabix_bed_path, chroms],
            check=True,
            text=True
        )

def log10_fdr_to_score(array):
    return np.round(array * 10).astype(np.int64).clip(0, 1000)


def norm_density_to_score(array):
    return np.round(array * 100).astype(np.int64).clip(0, 1000)