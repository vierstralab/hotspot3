import numpy as np
import numpy.ma as ma
import pandas as pd
from genome_tools.data.extractors import TabixExtractor, ChromParquetExtractor
from genome_tools.genomic_interval import GenomicInterval

from hotspot3.models import NoContigPresentError, ProcessorConfig
from hotspot3.fit import WindowBackgroundFit

counts_dtype = np.int32


class ChromosomeExtractor:
    def __init__(self, chrom_name: str, chrom_size: int, config: ProcessorConfig=None):
        self.chrom_name = chrom_name
        self.chrom_size = chrom_size
        self.genomic_interval = GenomicInterval(chrom_name, 0, chrom_size)
        if config is None:
            config = ProcessorConfig()
        self.fit_model = WindowBackgroundFit(config)

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
                raise NoContigPresentError
    
        return mappable
    
    def extract_cutcounts(self, cutcounts_file):
        cutcounts = np.zeros(self.chrom_size, dtype=np.int32)
        try:
            with TabixExtractor(cutcounts_file) as cutcounts_loader:
                data = cutcounts_loader[self.genomic_interval]
                assert data.eval('end - start == 1').all(), "Cutcounts are expected to be at basepair resolution"
                cutcounts[data['start']] = data['count'].to_numpy()
        except ValueError:
            raise NoContigPresentError

        return cutcounts
    
    def extract_aggregated_cutcounts(self, cutcounts_file):
        cutcounts = self.extract_cutcounts(cutcounts_file).astype(np.float32)
        window = self.fit_model.config.window
        
        agg_cutcounts = self.fit_model.centered_running_nansum(cutcounts, window, min_count=1)
        return agg_cutcounts
    

    def extract_mappable_agg_cutcounts(self, cutcounts_file, mappable_file):
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
            raise NoContigPresentError
        return signal_df

    def extract_fdr_track(self, fdr_path):
        log10_fdrs = self.extract_from_parquet(fdr_path, columns=['log10_fdr'])['log10_fdr'].values
        return log10_fdrs
