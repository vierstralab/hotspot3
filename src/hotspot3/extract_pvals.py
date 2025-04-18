import pandas as pd
import argparse
import numpy as np
import sys

from hotspot3.io.readers import GenomeReader, ChromReader
from genome_tools import GenomicInterval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pvals_parquet', type=str)
    parser.add_argument('bed', type=str)
    parser.add_argument("--chrom_sizes", help="Path to chromosome sizes file. If none assumed to be hg38 sizes", default=None)

    parser.add_argument('save_path', type=str)

    return parser.parse_args()

def extract_max_pval(row, chrom_pvals):
    return np.nanmax(chrom_pvals[row['start']:row['end']])

def main():
    args = parse_args()
    reader = GenomeReader()
    chrom_sizes = reader.read_chrom_sizes(args.chrom_sizes)
    if args.bed == '-':
        input_bed = sys.stdin
    else:
        input_bed = args.bed
    bed_df = pd.read_table(input_bed, header=None, names=['chrom', 'start', 'end'], comment='#', usecols=[0, 1, 2])
    groups = bed_df.groupby('chrom')
    data = []
    for chrom, group in groups:
        if chrom not in chrom_sizes:
            continue
        chrom_interval = GenomicInterval(chrom, 0, chrom_sizes[chrom])
        chrom_reader = ChromReader(genomic_interval=chrom_interval)
        chrom_pvals = chrom_reader.extract_from_parquet(
            args.pvals_parquet,
            columns=['log10_pval']
        )['log10_pval'].values
        group['max_neglog_p'] = group.apply(extract_max_pval, chrom_pvals=chrom_pvals, axis=1)
        data.append(group)

    data = pd.concat(data, ignore_index=True)
    data.sort_values(['chrom', 'start']).to_csv(
        args.save_path,
        sep='\t',
        index=False,
        header=None
    )

if __name__ == '__main__':
    main()

        


