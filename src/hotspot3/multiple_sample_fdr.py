from hotspot3.scoring.fdr import MultiSampleFDRCorrection
from hotspot3.io.logging import setup_logger

import pandas as pd
import logging
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mapping_df', type=str)
    parser.add_argument('--fdr_cutoff', type=float, default=0.05)
    parser.add_argument('save_path', type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    mapping_df = pd.read_table(args.mapping_df).set_index('id')['pvals_parquet']
    ms_fdr = MultiSampleFDRCorrection(
        name=mapping_df.index,
        logger=setup_logger(level=logging.DEBUG)
    )
    ms_fdr.fdr_correct_pvals(
        paths=mapping_df.to_dict(),
        fdr_cutoff=args.fdr_cutoff,
        save_path=args.save_path
    )