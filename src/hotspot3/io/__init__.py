import pandas as pd
import subprocess
import importlib.resources as pkg_resources


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
