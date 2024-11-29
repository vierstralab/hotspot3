import os
import shutil
import pandas as pd
import tempfile
import pyBigWig
import numpy as np
import subprocess

from genome_tools.helpers import df_to_tabix

from hotspot3.helpers.models import ProcessorOutputData, NotEnoughDataForContig, WindowedFitResults

from hotspot3.io import parallel_write_partitioned_parquet
from hotspot3.io.logging import WithLoggerAndInterval, WithLogger
from hotspot3.signal_smoothing import normalize_density


class ChromWriter(WithLoggerAndInterval):

    def parallel_write_chromdata_to_parquet(self, data_df, path, chrom_names, compression_level=22):
        """
        Workaround for writing parquet files for chromosomes in parallel.
        """
        chrom_name = self.genomic_interval.chrom
        if data_df is None:
            raise NotEnoughDataForContig
        if isinstance(data_df, ProcessorOutputData):
            data_df = data_df.data_df
        data_df['chrom'] = pd.Categorical(
            [chrom_name] * data_df.shape[0],
            categories=chrom_names
        )
        parallel_write_partitioned_parquet(
            data_df,
            field_names=[chrom_name],
            partition_cols=['chrom'],
            path=path,
            tmp_dir=self.config.tmp_dir,
            compression_level=compression_level
        )


class GenomeWriter(WithLogger):

    def df_to_bigwig(self, df: pd.DataFrame, outpath: str, chrom_sizes: dict, col='value'):
        with pyBigWig.open(outpath, 'w') as bw:
            bw.addHeader(list(chrom_sizes.items()))
            chroms = df['chrom'].to_list()
            starts = df['start'].to_list()
            ends = df['end'].to_list()
            values = df[col].to_list()
            bw.addEntries(chroms, starts, ends=ends, values=values)
    
    def sanitize_path(self, path):
        """
        Call to properly clean up the path to replace parquets
        """
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    
    def df_to_tabix(self, df, outpath):
        df_to_tabix(df, outpath)
    

    def thresholds_df_to_bw(self, thresholds: pd.DataFrame, save_path, total_cutcounts, chrom_sizes):
        thresholds['end'] = thresholds['start'] + self.config.bg_track_step
        thresholds['tr'] = normalize_density(thresholds['tr'], total_cutcounts)
        self.df_to_bigwig(
            thresholds,
            save_path,
            chrom_sizes=chrom_sizes,
            col='tr'
        )

    def density_to_bw(self, density_data: pd.DataFrame, save_path, chrom_sizes):
        density_data['end'] = density_data['start'] + self.config.density_track_step
        self.logger.debug(f"Converting density to bigwig")
        self.df_to_bigwig(
            density_data,
            save_path,
            chrom_sizes=chrom_sizes,
            col='normalized_density'
        )

    def fit_stats_to_bw(
            self,
            fit_stats: pd.DataFrame,
            outpath_bw,
            total_cutcounts,
            chrom_sizes
        ):
        fit_stats = fit_stats.query('fit_type == "segment"')[
            ['chrom', 'start', 'end', 'background']
        ]
        fit_stats['background'] = normalize_density(
            fit_stats['background'],
            total_cutcounts
        )

        self.df_to_bigwig(
            fit_stats,
            outpath_bw,
            chrom_sizes=chrom_sizes,
            col='background'
        )
    
    def save_cutcounts(self, total_cutcounts, total_cutcounts_path):
        np.savetxt(total_cutcounts_path, [total_cutcounts], fmt='%d')
    
    def df_to_bigbed(self, df: pd.DataFrame, chrom_sizes, outpath):
        with tempfile.NamedTemporaryFile(suffix=".bed") as temp_sorted_bed:
            df.to_csv(temp_sorted_bed.name, sep='\t', header=False, index=False)
            try:
                subprocess.run(["bedToBigBed", temp_sorted_bed.name, chrom_sizes, outpath], check=True)
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"Error converting to BigBed: {e}")

    def merge_partitioned_parquets(self, parquet_old, parquet_new):
        for file in os.listdir(parquet_old):
            new_path = os.path.join(parquet_new, file)
            if not os.path.exists(new_path):
                shutil.move(os.path.join(parquet_old, file), new_path)
        
        if self.config.save_debug:
            shutil.move(parquet_old, f"{parquet_old}.iter1")
        else:
            shutil.rmtree(parquet_old)
        shutil.move(parquet_new, parquet_old)