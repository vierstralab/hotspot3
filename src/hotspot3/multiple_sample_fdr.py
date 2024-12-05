from hotspot3.scoring.fdr import MultiSampleFDRCorrection
from hotspot3.io.logging import setup_logger
from hotspot3.config import ProcessorConfig
from hotspot3.io.readers import GenomeReader

import pandas as pd
import logging
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mapping_df', type=str)
    parser.add_argument('save_path', type=str)
    parser.add_argument('--fdr_cutoff', type=float, default=0.05)
    parser.add_argument('--chrom_sizes', type=str, default=None)
    parser.add_argument('--cpus', type=int, default=10)

    return parser.parse_args()

def main():
    args = parse_args()
    config = ProcessorConfig(cpus=args.cpus, logger_level=logging.DEBUG)
    reader = GenomeReader(config=config)
    chrom_sizes = reader.read_chrom_sizes(args.chrom_sizes)
    mapping_df = pd.read_table(args.mapping_df).set_index('id')['pvals_parquet']
    ms_fdr = MultiSampleFDRCorrection(
        name=mapping_df.index.tolist(),
        config=config,
        chrom_sizes=chrom_sizes
    )
    ms_fdr.fdr_correct_pvals(
        paths=mapping_df.to_dict(),
        fdr_cutoff=args.fdr_cutoff,
        save_path=args.save_path
    )