from hotspot3.scoring.fdr import MultiSampleFDRCorrection

import sys
import pandas as pd


def main(mapping_df: pd.Series, fdr_cutoff: float, save_path: str):
    ms_fdr = MultiSampleFDRCorrection(
        name=mapping_df.index,
    )
    ms_fdr.fdr_correct_pvals(
        paths=mapping_df.to_dict(),
        fdr_cutoff=fdr_cutoff,
        save_path=save_path
    )

if __name__ == "__main__":
    mapping_df = pd.read_table(sys.argv[1]).set_index('id')['pvals_parquet']
    fdr = float(sys.argv[2])
    save_path = sys.argv[3]
    main(mapping_df, fdr, save_path)