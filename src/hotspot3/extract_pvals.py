import pandas as pd
import argparse
import numpy as np
import sys
from tqdm import tqdm

from hotspot3.io.readers import GenomeReader, ChromReader
from hotspot3.helpers.models import NotEnoughDataForContig
from genome_tools import GenomicInterval

try:
    import numba

except ImportError:
    print("Numba is not installed. Please install it to use this script.")
    sys.exit(1)


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
    parser.add_argument('pvals_parquet', type=str, help='Path to parquet file with pvals')
    parser.add_argument('bed', type=str, help='Path to bed file with regions to extract pvals from. Use "-" for stdin')
    parser.add_argument('--format', choices=['bed', 'txt', 'npy'], default='bed', help='Output format')
    parser.add_argument("--chrom_sizes", help="Path to chromosome sizes file. If none assumed to be hg38 sizes", default=None)

    parser.add_argument('save_path', type=str, help='Path to save results')

    return parser.parse_args()

def extract_max_pval(row, chrom_pvals):
    return np.nanmax(chrom_pvals[row['start']:row['end']], initial=0)

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
            max_pvals = np.zeros(len(group), dtype=np.float32)
        else:
            chrom_interval = GenomicInterval(chrom, 0, chrom_sizes[chrom])

            chrom_reader = ChromReader(genomic_interval=chrom_interval)
            try:
                chrom_pvals = chrom_reader.extract_from_parquet(
                    args.pvals_parquet,
                    columns=['log10_pval']
                )['log10_pval'].values
                max_pvals = group.apply(extract_max_pval, chrom_pvals=chrom_pvals, axis=1)
            except NotEnoughDataForContig:
                max_pvals = np.zeros(len(group), dtype=np.float32)
            
        group['max_neglog_p'] = max_pvals
        data.append(group)

    data = pd.concat(data).sort_index()
    if args.format == 'npy':
        np.save(args.save_path, data['max_neglog_p'].astype(np.float32).values)
    elif args.format == "txt":
        np.savetxt(
            args.save_path,
            data['max_neglog_p'].astype(np.float32).values,
        )
    else:
        data.to_csv(
            args.save_path,
            sep='\t',
            index=False,
            header=None
        )

if __name__ == '__main__':
    main()

        


