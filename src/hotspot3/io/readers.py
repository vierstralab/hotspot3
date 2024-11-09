import numpy as np
import numpy.ma as ma
import pandas as pd
from genome_tools.data.extractors import TabixExtractor, ChromParquetExtractor
from genome_tools.genomic_interval import GenomicInterval

from hotspot3.models import NotEnoughDataForContig, WindowedFitResults
from hotspot3.config import ProcessorConfig
from hotspot3.connectors.bottleneck import BottleneckWrapper
from hotspot3.io.logging import WithLoggerAndInterval, WithLogger

counts_dtype = np.int32


class ChromReader(WithLoggerAndInterval):
    def __init__(self, genomic_interval: GenomicInterval, config: ProcessorConfig=None, logger=None):
        super().__init__(genomic_interval=genomic_interval, config=config, logger=logger)

        self.chrom_name = genomic_interval.chrom
        self.chrom_size = genomic_interval.end

        self.bottleneck_model = BottleneckWrapper(config)

    def extract_mappable_bases(self, mappable_file) -> np.ndarray:
        """
        Extract mappable bases for the chromosome.
        """
        if mappable_file is None:
            mappable = np.ones(self.chrom_size, dtype=bool)
        else:
            mappable = np.zeros(self.chrom_size, dtype=bool)
            try:
                with TabixExtractor(mappable_file, columns=['#chr', 'start', 'end']) as mappable_loader:
                    for _, row in mappable_loader[self.genomic_interval].iterrows():
                        if row['end'] > self.genomic_interval.end:
                            raise ValueError(f"Mappable bases file does not match chromosome sizes! Check input parameters. {row['end']} > {self.genomic_interval.end} for {self.chrom_name}")
                        mappable[row['start']:row['end']] = True
            except ValueError:
                raise NotEnoughDataForContig
    
        return mappable
    
    def extract_cutcounts(self, cutcounts_file):
        cutcounts = np.zeros(self.chrom_size, dtype=np.int32)
        try:
            with TabixExtractor(cutcounts_file) as cutcounts_loader:
                data = cutcounts_loader[self.genomic_interval]
                assert data.eval('end - start == 1').all(), "Cutcounts are expected to be at basepair resolution"
                cutcounts[data['start']] = data['count'].to_numpy()
        except ValueError:
            raise NotEnoughDataForContig

        return cutcounts
    
    def extract_aggregated_cutcounts(self, cutcounts_file):
        cutcounts = self.extract_cutcounts(cutcounts_file).astype(np.float32)
        window = self.bottleneck_model.config.window
        
        agg_cutcounts = self.bottleneck_model.centered_running_nansum(cutcounts, window)
        return agg_cutcounts
    

    def extract_mappable_agg_cutcounts(self, cutcounts_file, mappable_file) -> ma.MaskedArray:
        agg_cutcounts = self.extract_aggregated_cutcounts(cutcounts_file)
        mappable = self.extract_mappable_bases(mappable_file)
        return ma.masked_where(~mappable, agg_cutcounts)

    
    def extract_from_parquet(self, signal_parquet, columns) -> pd.DataFrame:
        with ChromParquetExtractor(
            signal_parquet,
            columns=columns
        ) as smoothed_signal_loader:
            signal_df = smoothed_signal_loader[self.genomic_interval]
        if signal_df.empty:
            raise NotEnoughDataForContig
        return signal_df
    
    def extract_fit_params(self, fit_params_parquet) -> WindowedFitResults:
        fit_params = self.extract_from_parquet(
            fit_params_parquet,
            columns=['sliding_r', 'sliding_p', 'enough_bg']
        )
        return WindowedFitResults(
            fit_params['sliding_p'].values,
            fit_params['sliding_r'].values,
            fit_params['enough_bg'].values
        )

    def extract_fdr_track(self, fdr_path):
        log10_fdrs = self.extract_from_parquet(fdr_path, columns=['log10_fdr'])['log10_fdr'].values
        return log10_fdrs


class GenomeReader(WithLogger):

    def read_full_parquet(self, pvals_path, column):
        return pd.read_parquet(
            pvals_path,
            engine='pyarrow', 
            columns=column
        )[column]

    def read_pval_from_parquet(self, pvals_path):
        return self.read_full_parquet(pvals_path, column='log10_pval').values
    
    def read_chrom_pos_mapping(self, pvals_path):
        chrom_pos_mapping = self.read_full_parquet(pvals_path, column='chrom')

        total_len = chrom_pos_mapping.shape[0]
        chrom_pos_mapping = chrom_pos_mapping.drop_duplicates()
        starts = chrom_pos_mapping.index
        # file is always sorted within chromosomes
        ends = [*starts[1:], total_len]
        return chrom_pos_mapping, starts, ends