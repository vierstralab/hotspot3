import os
import shutil
import pandas as pd
import tempfile
import pyBigWig
import numpy as np
import subprocess

from genome_tools.helpers import df_to_tabix

from hotspot3.io.logging import WithLoggerAndInterval, WithLogger
from hotspot3.models import ProcessorOutputData, NotEnoughDataForContig
from hotspot3.io import to_parquet_high_compression, log10_fdr_to_score, norm_density_to_score
from hotspot3.io.colors import get_bb_color


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


    bed12_columns = ['chrom', 'start', 'end', 'id', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']


    def df_to_bigwig(self, df: pd.DataFrame, outpath, chrom_sizes: dict, col='value'):
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
    
    def df_to_gzip(self, df: pd.DataFrame, outpath):
        df.to_csv(outpath, sep='\t', index=False, compression='gzip')
    
    def save_cutcounts(self, total_cutcounts, total_cutcounts_path):
        np.savetxt(total_cutcounts_path, [total_cutcounts], fmt='%d')
    
    def df_to_bigbed(self, df: pd.DataFrame, chrom_sizes, outpath):
        with tempfile.NamedTemporaryFile(suffix=".bed") as temp_sorted_bed:
            # Write sorted data to the temporary file
            df.to_csv(temp_sorted_bed.name, sep='\t', header=False, index=False)
            
            # Run bedToBigBed
            try:
                subprocess.run(["bedToBigBed", temp_sorted_bed.name, chrom_sizes, outpath], check=True)
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"Error converting to BigBed: {e}")
    

    def peaks_to_bed12(self, peaks_df, fdr_tr):
        """
        Convert peaks to bed9 format.
        """
        peaks_df['strand'] = '.'
        peaks_df['score'] = norm_density_to_score(peaks_df['max_density'])
        peaks_df['thickStart'] = peaks_df['summit']
        peaks_df['thickEnd'] = peaks_df['summit'] + 1
        peaks_df['itemRgb'] = get_bb_color(fdr_tr, mode='peaks')

        peaks_df['blockCount'] = 3
        peaks_df['blockSizes'] = '1,1,1'
        peaks_df['blockStarts'] = '0,' + peaks_df.eval('summit - start').astype(str) + ',' + peaks_df.eval('end - start - 1').astype(str)

        return peaks_df[self.bed12_columns]


    def hotspots_to_bed12(self, hotspots_df, fdr_tr, significant_stretches):
        """
        Convert hotspots to bed9 format.
        """
        hotspots_df['strand'] = '.'
        hotspots_df['score'] = log10_fdr_to_score(hotspots_df['max_neglog10_fdr'])
        hotspots_df['thickStart'] = hotspots_df['start']
        hotspots_df['thickEnd'] = hotspots_df['end']
        hotspots_df['itemRgb'] = get_bb_color(fdr_tr, mode='hotspots')
        block_count = []
        block_sizes = []
        block_starts = []
        lengths = hotspots_df.eval('end - start').values
        for i, (starts, ends) in enumerate(significant_stretches):
            block_count.append(len(starts) + 2)

            sizes = np.pad(ends - starts, (1, 1), mode='constant', constant_values=(1, 1))
            block_sizes.append(','.join(map(str, sizes)))

            starts = np.pad(starts, (1, 1), mode='constant', constant_values=(0, lengths[i] - 1))
            block_starts.append(','.join(map(str, starts)))

        hotspots_df['blockCount'] = block_count
        hotspots_df['blockSizes'] = block_sizes
        hotspots_df['blockStarts'] = block_starts

        return hotspots_df[self.bed12_columns]

    def get_chrom_sizes_file(self, chrom_sizes_file):
        if chrom_sizes_file is None:
            raise NotImplementedError("Chromosome sizes file is not embedded yet")
        return chrom_sizes_file
