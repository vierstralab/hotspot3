"""
This script is used to produce q-values corrected for the set of samples
"""
from hotspot3.io.logging import WithLogger
from hotspot3.io.readers import GenomeReader
from hotspot3.io.writers import GenomeWriter
from hotspot3.io import parallel_write_partitioned_parquet, read_partioned_parquet

from hotspot3.scoring import find_potentialy_significant_pvals, logfdr_from_logpvals
from hotspot3.helpers.models import SampleFDRdata, MultiSampleFDRData, FDRData, NotEnoughDataForContig
from hotspot3.config import ProcessorConfig

import numpy as np
import pandas as pd
import shutil
from logging import Logger
import gc
from concurrent.futures import ProcessPoolExecutor
from typing import List
from genome_tools import GenomicInterval


class FDRCorrection(WithLogger):

    def __init__(self, name, config: ProcessorConfig=None, logger: Logger=None, *, chrom_sizes: dict=None):
        super().__init__(config, logger, name)
        self.chrom_sizes = chrom_sizes
        self.reader = self.copy_with_params(GenomeReader)
        self.writer = self.copy_with_params(GenomeWriter)

    def compute_fdr(self, fdr_data: FDRData) -> np.ndarray:
        potentially_signif_pvals = fdr_data.potentially_signif_pvals
        potentially_signif_pvals *= -np.log(10)

        logfdr = logfdr_from_logpvals(
            potentially_signif_pvals,
            method=self.config.fdr_method,
            m=fdr_data.n_tests
        ).astype(np.float16)
        logfdr /= -np.log(10)
        
        return logfdr
    
    def fdr_correct_pvals(self):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    def extract_data_for_sample(self):
        raise NotImplementedError("This method should be implemented in a subclass")


class SampleFDRCorrection(FDRCorrection):

    def fdr_correct_pvals(self, pvals_path, max_fdr, save_path):

        mask_path = f"{save_path}.mask"
        self.writer.sanitize_path(mask_path)
        self.logger.debug(f"{self.name}: Extracting raw P-values")
        fdr_data, mask = self.extract_data_for_sample(pvals_path, max_fdr, mask_path, return_mask=True)

        self.logger.debug(f"{self.name}: Computing FDR")
        result = self.compute_fdr(fdr_data)
        result = self.cast_to_original_shape(result, mask)

        self.writer.sanitize_path(save_path)
        self.logger.debug(f"{self.name}: Saving FDRs to parquet")
        self.write_fdr_partitioned_by_sample_and_chrom(
            fdr_data.chrom_pos_mapping,
            result,
            save_path,
            cpus=self.config.cpus
        )
    
    def read_fdrs_for_chrom(self, fdr_path, chrom):
        return read_partioned_parquet(
            fdr_path,
            partition_cols=['chrom', 'sample_id'],
            partition_vals=[chrom, self.name],
            columns=['log10_fdr']
        )['log10_fdr'].values


    def extract_data_for_sample(self, pvals_path, fdr, save_path, all_ids=None, return_mask=False):
        if all_ids is None:
            all_ids = [self.name]
        log_pvals = self.reader.read_pval_from_parquet(pvals_path)
        mask, n_tests = find_potentialy_significant_pvals(log_pvals, fdr)
        log_pvals = log_pvals[mask]

        chrom_pos_mapping = self.reader.read_chrom_pos_mapping(
            pvals_path,
            chrom_sizes=self.chrom_sizes,
        )
        data = SampleFDRdata(log_pvals, n_tests, self.name, chrom_pos_mapping)
        if return_mask:
            return data, mask
        self.logger.debug(f"{self.name}: Saving mask to parquet")
        mask = pd.DataFrame({
            'tested_pos': mask,
            'sample_id': pd.Categorical([self.name] * len(mask), categories=all_ids),
        })
        self.write_partitioned_by_sample_df_to_parquet(mask, save_path)
        return data
            
    
    def write_partitioned_by_sample_df_to_parquet(self, df: pd.DataFrame, save_path):
        parallel_write_partitioned_parquet(
            df,
            [self.name],
            partition_cols=['sample_id'],
            path=save_path,
            tmp_dir=self.config.tmp_dir,
        )
    
    def extract_mask_for_sample(self, mask_path) -> np.ndarray:
        return read_partioned_parquet(
            mask_path,
            partition_col=['sample_id'],
            partition_val=[self.name],
            columns=['tested_pos']
        )['tested_pos'].values
    
    def cast_to_original_shape(self, result: np.ndarray, mask) -> np.ndarray:
        if isinstance(mask, str):
            mask = self.extract_mask_for_sample(mask)
        res = np.full(mask.shape, np.nan, dtype=result.dtype)
        res[mask] = result
        return res
    
    def write_fdr_partitioned_by_sample_and_chrom(
            self,
            chrom_pos_mapping: List[GenomicInterval],
            fdrs: np.ndarray,
            save_path,
            all_ids=None,
            cpus=None
        ):
        if all_ids is None:
            all_ids = [self.name]
        
        if cpus is None:
            cpus = self.config.cpus

        if cpus == 1:
            for gi in chrom_pos_mapping:
                chrom_df = self.df_from_fdrs(fdrs, gi, all_ids)
                parallel_write_partitioned_parquet(
                    chrom_df,
                    field_names=[gi.chrom, self.name],
                    partition_cols=['chrom', 'sample_id'],
                    path=save_path,
                    tmp_dir=self.config.tmp_dir
                )
        else:
            params = [
                (
                    self.df_from_fdrs(fdrs, gi, all_ids),
                    (gi.chrom, self.name),
                    ['chrom', 'sample_id'],
                    save_path,
                    self.config.tmp_dir
                )
                for gi in chrom_pos_mapping
            ]
            params = [list(x) for x in zip(*params)]
            with ProcessPoolExecutor(cpus) as executor:
                [x for x in executor.map(parallel_write_partitioned_parquet, *params)]
        
    def df_from_fdrs(self, fdrs, genomic_interval: GenomicInterval, all_ids: List[str]) -> pd.DataFrame:
        sample_id_codes = np.full(len(genomic_interval), all_ids.index(self.name), dtype=np.int16)
        
        chrom_categories = list(self.chrom_sizes.keys())
        chrom_codes = np.full(
            len(genomic_interval),
            chrom_categories.index(genomic_interval.chrom),
            dtype=np.int16
        )
        return pd.DataFrame({
            'log10_fdr': fdrs[genomic_interval.start:genomic_interval.end],
            'chrom': pd.Categorical.from_codes(chrom_codes, categories=chrom_categories),
            'sample_id': pd.Categorical.from_codes(sample_id_codes, categories=all_ids)
        })


class MultiSampleFDRCorrection(FDRCorrection):
    def fdr_correct_pvals(self, paths, fdr_cutoff, save_path):
        if fdr_cutoff > 0.05:
            self.logger.warning(f"Requested FDR is higher than 0.05: {fdr_cutoff}. The code is not optimized for high FDRs with high number of samples")

        mask_path = f"{save_path}.mask"
        self.writer.sanitize_path(mask_path)
        all_samples_fdr_data = self.extract_data_for_sample(paths, fdr_cutoff, mask_path)
        
        logfdr = self.compute_fdr(all_samples_fdr_data)

        all_samples_fdr_data.potentially_signif_pvals = None
        gc.collect()
        
        self.writer.sanitize_path(save_path)
        self.write_fdr_partitioned_by_sample_and_chrom(
            all_samples_fdr_data,
            logfdr,
            mask_path,
            save_path
        )
        shutil.rmtree(mask_path) # cleanup


    def extract_data_for_sample(self, paths: dict, fdr, save_path):
        results = []
        chrom_pos_mappings = []
        n_tests = 0
        sample_id_correspondance = pd.DataFrame(
            {'start_index': pd.NA, 'end_index': pd.NA},
            index=self.name
        )
        current_index = 0
        self.writer.sanitize_path(save_path)

        if self.config.cpus > 0:
            for sample_id, pvals_path in paths.items():
                self.logger.debug(f"Extracting data for {sample_id}")
                sample_correction = self.copy_with_params(
                    SampleFDRCorrection,
                    name=sample_id
                )
                fdr_correction_data = sample_correction.extract_data_for_sample(
                    pvals_path,
                    fdr,
                    all_ids=self.name,
                    save_path=save_path
                )
                potentially_significant_pvals = fdr_correction_data.potentially_signif_pvals
                n_tests += fdr_correction_data.n_tests

                sample_id_correspondance.loc[sample_id, 'start_index'] = current_index
                current_index += len(potentially_significant_pvals)
                sample_id_correspondance.loc[sample_id, 'end_index'] = current_index
                chrom_pos_mappings.append(fdr_correction_data.chrom_pos_mapping)
   
                results.append(potentially_significant_pvals)
        self.logger.debug(f"Data extracted for {len(paths)} samples")
        sample_id_correspondance = sample_id_correspondance.astype(int)
        potentially_significant_pvals = np.concatenate(results)

        return MultiSampleFDRData(
            potentially_significant_pvals,
            n_tests,
            sample_id_correspondance,
            chrom_pos_mappings
        )

    def write_fdr_partitioned_by_sample_and_chrom(
            self,
            all_samples_fdr_data: MultiSampleFDRData,
            logfdr: np.ndarray,
            mask_path,
            save_path
        ):
        sample_id_correspondance = all_samples_fdr_data.sample_id_correspondance
        for i, (sample_id, row) in enumerate(sample_id_correspondance.iterrows()):
            self.logger.debug(f"Writing FDR for {sample_id}")
            sample_correction = self.copy_with_params(
                SampleFDRCorrection,
                identifier=sample_id,
            )
            sample_fdrs = logfdr[row['start_index']:row['end_index']]
            sample_fdrs = sample_correction.cast_to_original_shape(sample_fdrs, mask_path)

            sample_correction.write_fdr_partitioned_by_sample_and_chrom(
                chrom_pos_mapping=all_samples_fdr_data.chrom_pos_mappings[i],
                fdrs=sample_fdrs,
                save_path=save_path,
                all_ids=self.name
            )