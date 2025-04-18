import pandas as pd
import argparse
import numpy as np
import sys
from tqdm import tqdm

from hotspot3.io.readers import GenomeReader, ChromReader
from genome_tools import GenomicInterval
import numba


@numba.njit
def segment_max(vec, starts, ends, out):
    for i in range(starts.shape[0]):
        m = vec[starts[i]]
        for j in range(starts[i]+1, ends[i]+1):
            if vec[j] > m:
                m = vec[j]
        out[i] = m


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pvals_parquet', type=str)
    parser.add_argument('bed', type=str)
    parser.add_argument("--chrom_sizes", help="Path to chromosome sizes file. If none assumed to be hg38 sizes", default=None)

    parser.add_argument('save_path', type=str)

    return parser.parse_args()


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
    for chrom, group in tqdm(groups, total=len(groups)):
        if chrom not in chrom_sizes:
            continue
        chrom_interval = GenomicInterval(chrom, 0, chrom_sizes[chrom])
        max_neglog_p = np.zeros(len(chrom_interval), dtype=np.float32)

        chrom_reader = ChromReader(genomic_interval=chrom_interval)
        chrom_pvals = chrom_reader.extract_from_parquet(
            args.pvals_parquet,
            columns=['log10_pval']
        )['log10_pval'].values.astype(np.float32)
        segment_max(chrom_pvals, group['start'].values, group['end'].values, max_neglog_p)
        group['max_neglog_p'] = max_neglog_p
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

        


