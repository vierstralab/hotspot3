import os
import shutil
import pandas as pd
import tempfile
import pyBigWig
import numpy as np
import subprocess

from genome_tools.helpers import df_to_tabix

from hotspot3.models import ProcessorOutputData, NotEnoughDataForContig, WindowedFitResults

from hotspot3.io import to_parquet_high_compression, convert_to_score
from hotspot3.io.logging import WithLoggerAndInterval, WithLogger
from hotspot3.io.colors import get_bb_color
from hotspot3.signal_smoothing import normalize_density


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
    
    def fit_results_to_df(self, fit_results: WindowedFitResults, per_window_trs: np.ndarray):
        return pd.DataFrame({
            'sliding_r': fit_results.r,
            'sliding_p': fit_results.p,
            'enough_bg': fit_results.enough_bg_mask,
            'tr': per_window_trs,
        })

    def update_fit_params(self, fit_params: WindowedFitResults, fit_results: WindowedFitResults):
        
        fit_params.r = np.where(
            fit_results.enough_bg_mask,
            fit_results.r,
            fit_params.r
        )
        fit_params.p = np.where(
            fit_results.enough_bg_mask,
            fit_results.p,
            fit_params.p
        )
        fit_params.enough_bg_mask = np.where(
            fit_results.enough_bg_mask,
            fit_results.enough_bg_mask,
            fit_params.enough_bg_mask
        )
        return fit_params

    def update_per_window_trs(self, initial_trs: np.ndarray, trs: np.ndarray, fit_results: WindowedFitResults):
        initial_trs[fit_results.enough_bg_mask] = trs[fit_results.enough_bg_mask]
        return initial_trs



class GenomeWriter(WithLogger):

    def __init__(self, *args, chrom_sizes, **kwargs):
        super().__init__(*args, **kwargs)
        self.chrom_sizes = chrom_sizes

    bed12_columns = ['chrom', 'start', 'end', 'id', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']


    def df_to_bigwig(self, df: pd.DataFrame, outpath, col='value'):
        with pyBigWig.open(outpath, 'w') as bw:
            bw.addHeader(list(self.chrom_sizes.items()))
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
    

    def thresholds_df_to_bw(self, thresholds: pd.DataFrame, save_path, total_cutcounts):
        thresholds['end'] = thresholds['start'] + self.config.bg_track_step
        thresholds['tr'] = normalize_density(thresholds['tr'], total_cutcounts)
        self.df_to_bigwig(
            thresholds,
            save_path,
            col='tr'
        )

    def density_to_bw(self, density_data: pd.DataFrame, save_path):
        density_data['end'] = density_data['start'] + self.config.density_track_step
        self.logger.debug(f"Converting density to bigwig")
        self.df_to_bigwig(
            density_data,
            save_path,
            col='normalized_density'
        )

    def fit_stats_to_tabix_and_bw(
            self,
            fit_stats: pd.DataFrame,
            outpath,
            outpath_bw,
            total_cutcounts,
        ):
        self.df_to_tabix(fit_stats, outpath)

        fit_stats = fit_stats.query('fit_type == "segment"')[['chrom', 'start', 'end', 'background']]
        fit_stats['background'] = normalize_density(
            fit_stats['background'],
            total_cutcounts
        )

        self.df_to_bigwig(
            fit_stats,
            outpath_bw,
            col='background'
        )
    
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
        peaks_df['score'] = convert_to_score(peaks_df['max_density'], 100)
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
        hotspots_df['score'] = convert_to_score(hotspots_df['max_neglog10_fdr'], 10)
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
