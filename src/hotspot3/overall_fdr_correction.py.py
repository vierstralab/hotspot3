"""
This script is used to produce q-values corrected for the set of samples
"""
from hotspot3.io.logging import WithLogger
from hotspot3.io.readers import GenomeReader
from hotspot3.io.writers import GenomeWriter
from hotspot3.io import parallel_write_partitioned_parquet, read_partioned_parquet

from hotspot3.scoring import find_potentialy_significant_pvals, logfdr_from_logpvals
from hotspot3.config import ProcessorConfig

import numpy as np
import pandas as pd
import shutil
from logging import Logger
import gc

import dataclasses
@dataclasses.dataclass
class FDRData:
    potentially_signif_pvals: np.ndarray
    n_tests: int

@dataclasses.dataclass
class SampleFDRdata(FDRData):
    sample_id: str


@dataclasses.dataclass
class MultiSampleFDRdata(FDRData):
    sample_id_correspondance: pd.DataFrame


class FDRCorrection(WithLogger):

    def __init__(self, name, config: ProcessorConfig=None, logger: Logger=None):
        super().__init__(config, logger, name)
        self.reader = self.copy_with_params(GenomeReader)
        self.writer = self.copy_with_params(GenomeWriter)

    def compute_fdr(self, fdr_data: FDRData):
        potentially_signif_pvals = fdr_data.potentially_signif_pvals
        potentially_signif_pvals *= -np.log(10)

        logfdr = logfdr_from_logpvals(
            potentially_signif_pvals,
            method=self.config.fdr_method,
            m=fdr_data.n_tests
        ).astype(np.float16) / -np.log(10)
        
        return logfdr


class OverallFDRCorrection(FDRCorrection):

    def fdr_correct_multiple_samples(self, paths, fdr, save_path):
        if fdr > 0.05:
            self.logger.warning(f"Requested FDR is higher than 0.05: {fdr}. The code is not optimized for high FDRs with high number of samples")

        mask_path = f"{save_path}.mask"
        self.writer.sanitize_path(mask_path)
        all_samples_fdr_data = self.extract_pval_data(paths, fdr, mask_path)
        
        logfdr = self.compute_fdr(all_samples_fdr_data)
        sample_id_correspondance = all_samples_fdr_data.sample_id_correspondance

        del all_samples_fdr_data
        gc.collect()
        
        self.writer.sanitize_path(save_path)
        self.write_fdr_to_parquet(logfdr, sample_id_correspondance, mask_path, save_path)
        shutil.rmtree(mask_path) # cleanup


    def write_fdr_to_parquet(
            self,
            logfdr: np.ndarray,
            sample_id_correspondance: pd.DataFrame,
            mask_path,
            save_path
        ):
        # save mask
        for sample_id, row in sample_id_correspondance.iterrows():
            sample_correction = self.copy_with_params(
                SampleFDRCorrection,
                identifier=sample_id,
            )
            mask = sample_correction.extract_mask_for_sample(mask_path)

            df = pd.DataFrame({
                'logfdr': logfdr[row['start_index']:row['end_index']],
                'sample_id': pd.Categorical(sample_id, categories=self.name),
                'start': np.arange(mask.shape[0], dtype=np.uint32)[mask]
            })
            sample_correction.write_partitioned_by_sample_df_to_parquet(df, save_path)


    def extract_pval_data(self, paths: dict, fdr, save_path):
        results = []
        n_tests = 0
        sample_id_correspondance = pd.DataFrame(
            {'start_index': pd.NA, 'end_index': pd.NA},
            index=self.name
        )
        self.writer.sanitize_path(save_path)
        for sample_id, pvals_path in paths.items():
            sample_correction = self.copy_with_params(
                SampleFDRCorrection,
                name=sample_id
            )
            fdr_correction_data = sample_correction.extract_data_for_sample(
                pvals_path,
                fdr,
                all_ids=self.all_ids,
                save_path=save_path
            )
            potentially_significant_pvals = fdr_correction_data.potentially_signif_pvals

            sample_id_correspondance.loc[sample_id, 'start_index'] = n_tests
            n_tests += fdr_correction_data.n_tests
            sample_id_correspondance.loc[sample_id, 'end_index'] = n_tests
   
            results.append(potentially_significant_pvals)
        
        sample_id_correspondance = sample_id_correspondance.astype(int)
        potentially_significant_pvals = np.concatenate(results)

        return MultiSampleFDRdata(potentially_significant_pvals, n_tests, sample_id_correspondance)


class SampleFDRCorrection(FDRCorrection):

    def fdr_correct_pvals(self, pvals_path, max_fdr, save_path):

        mask_path = f"{save_path}.mask"
        fdr_data = self.extract_data_for_sample(pvals_path, max_fdr, mask_path)

        log_fdr = self.compute_fdr(fdr_data)

        self.writer.sanitize_path(save_path)
        self.write_fdr_partitioned_by_sample_and_chrom(log_fdr, save_path)


    def extract_data_for_sample(self, pvals_path, fdr, save_path, all_ids=None):
        if all_ids is None:
            all_ids = [self.name]
        log_pvals = self.reader.read_pval_from_parquet(pvals_path)
        mask, n_tests = find_potentialy_significant_pvals(log_pvals, fdr)
        log_pvals = log_pvals[mask]
        mask = pd.DataFrame({
            'tested_pos': mask,
            'sample_id': pd.Categorical(self.name, categories=all_ids),
        })
        
        self.write_partitioned_by_sample_df_to_parquet(mask, save_path)
        return SampleFDRdata(self.name, log_pvals, n_tests)
    
    def write_partitioned_by_sample_df_to_parquet(self, df: pd.DataFrame, save_path):
        parallel_write_partitioned_parquet(
            df,
            self.name,
            partition_col='sample_id',
            path=save_path,
            tmp_dir=self.config.tmp_dir,
        )
    
    def extract_mask_for_sample(self, mask_path):
        return read_partioned_parquet(
            mask_path,
            partition_col='sample_id',
            partition_val=self.name,
            columns=['tested_pos']
        )['tested_pos'].values
    
    def write_fdr_partitioned_by_sample_and_chrom(self, df: pd.DataFrame, save_path):
        parallel_write_partitioned_parquet(
            df,
            partition_col='sample_id',
            path=save_path,
            tmp_dir=self.config.tmp_dir
        )
