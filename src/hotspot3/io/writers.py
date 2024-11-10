import os
import shutil
import pandas as pd
import tempfile
import pyBigWig
import numpy as np

from genome_tools.helpers import df_to_tabix

from hotspot3.io.logging import WithLoggerAndInterval, WithLogger
from hotspot3.models import ProcessorOutputData, NotEnoughDataForContig
from hotspot3.io import to_parquet_high_compression


class ChromWriter(WithLoggerAndInterval):

    def parallel_write_to_parquet(self, data_df, path, chrom_names, compression_level=22):
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
        os.makedirs(path, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.config.tmp_dir) as temp_dir:
            temp_path = os.path.join(temp_dir, f'{chrom_name}.temp.parquet')
            to_parquet_high_compression(
                data_df,
                temp_path,
                compression_level=compression_level
            )
            res_path = os.path.join(path, f'chrom={chrom_name}')
            if os.path.exists(res_path):
                shutil.rmtree(res_path)
            shutil.move(os.path.join(temp_path, f'chrom={chrom_name}'), path)


class GenomeWriter(WithLogger):
    def df_to_bigwig(self, df: pd.DataFrame, outpath, chrom_sizes: dict, col='value'):
        with pyBigWig.open(outpath, 'w') as bw:
            bw.addHeader(list(chrom_sizes.items()))
            chroms = df['chrom'].to_list()
            starts = df['start'].to_list()
            ends = df['end'].to_list()
            values = df[col].to_list()
            bw.addEntries(chroms, starts, ends=ends, values=values)
    
    def clean_path(self, path):
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
    
    def df_to_gzip(self, df: pd.DataFrame, outpath):
        df.to_csv(outpath, sep='\t', index=False, compression='gzip')
    
    def save_cutcounts(total_cutcounts, total_cutcounts_path):
        np.savetxt(total_cutcounts_path, [total_cutcounts], fmt='%d')
