import logging
import numpy as np
import numpy.ma as ma
from scipy.signal import convolve
import scipy.stats as st
from functools import reduce
from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from genome_tools.genomic_interval import GenomicInterval
from genome_tools.data.extractors import TabixExtractor
import multiprocessing as mp
import pandas as pd
import dataclasses
from statsmodels.stats.multitest import multipletests
from typing import Tuple
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
root_logger = logging.getLogger(__name__)

def set_logger_config(logger, level):
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)


@dataclasses.dataclass
class PeakCallingData:
    chrom: str
    data_df: pd.DataFrame
    params_df: pd.DataFrame


class NoContigPresentError(Exception):
    ...


class GenomeProcessor:
    """
    Base class to run hotspot2 on a whole genome
    """
    def __init__(self, chrom_sizes, mappable_bases_file=None, window=201, bg_window=50001, min_mappable_bg=10000, signal_tr=0.975, int_dtype = np.int32, fdr_method='fdr_bh', cpus=1, chromosomes=None, logger_level=logging.DEBUG) -> None:
        self.logger = root_logger
        self.logger_level = logger_level
        self.chrom_sizes = chrom_sizes
        if chromosomes is not None:
            self.chrom_sizes = {k: v for k, v in chrom_sizes.items() if k in chromosomes}
        self.mappable_bases_file = mappable_bases_file
        self.min_mappable_bg = min_mappable_bg

        self.bg_window = bg_window
        self.window = window
        self.int_dtype = int_dtype
        
        self.cpus = min(cpus, max(1, mp.cpu_count()))
        self.signal_tr = 1 - signal_tr
        self.fdr_method = fdr_method

        self.chromosome_processors = [x for x in self.get_chromosome_processors()]
        self.logger.info(f"Chromosomes with mappable track: {len(self.chromosome_processors)}")
    
    def __getstate__(self):
        state = self.__dict__
        if 'logger' in state:
            del state['logger']
        return state

    def restore_logger(self):
        self.logger = root_logger
        set_logger_config(self.logger, self.logger_level)

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)
        assert not hasattr(self, 'logger')
        self.restore_logger()

    def get_chromosome_processors(self):
        for chrom_name in self.chrom_sizes.keys():
            try:
                yield ChromosomeProcessor(self, chrom_name)
            except NoContigPresentError:
                continue
        
    def call_peaks(self, cutcounts_file) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.debug(f'Using {self.cpus} CPUs')
        if self.cpus == 1:
            result = []
            for chrom_processor in self.chromosome_processors:
                res = chrom_processor.calc_pvals(cutcounts_file)
                result.append(res)
    
        else:
            ctx = mp.get_context("forkserver")
            with ctx.Pool(self.cpus) as executor:
                results = executor.starmap(
                    ChromosomeProcessor.calc_pvals,
                    [(cp, cutcounts_file) for cp in self.chromosome_processors])
    
        self.restore_logger()
        self.logger.debug('Concatenating results')
        print([result for result in results])
        data_df = pd.concat([result.data_df for result in results])
        data_df['fdr'] = self.calc_fdr(data_df['log10_pval'])

        params_df = pd.concat([result.params_df for result in results])
        return data_df[['#chr', 'start', 'log10_pval', 'fdr']], params_df

    def calc_fdr(self, pval_list):
        fdr = np.empty(pval_list.shape)
        fdr[pval_list.mask] = np.nan
        fdr[~pval_list.mask] = multipletests(pval_list.compressed(), method=self.fdr_method)[1]
        return fdr


class ChromosomeProcessor:
    def __init__(self, genome_processor: GenomeProcessor, chrom_name: str) -> None:
        self.chrom_name = chrom_name
        self.gp = genome_processor
        self.chrom_size = self.gp.chrom_sizes[chrom_name]
        self.genomic_interval = GenomicInterval(chrom_name, 0, self.chrom_size)
  
        self.int_dtype = self.gp.int_dtype
        self.mappable_bases = self.get_mappable_bases(self.gp.mappable_bases_file)

    def calc_pvals(self, cutcounts_file) -> PeakCallingData:
        self.gp.logger.debug(f'Extracting cutcounts for chromosome {self.chrom_name}')
        cutcounts = self.extract_cutcounts(cutcounts_file)
        self.gp.logger.debug(f'Aggregating cutcounts for chromosome {self.chrom_name}')
        
        agg_cutcounts = self.smooth_counts(cutcounts, self.gp.window)
        agg_cutcounts_masked = np.ma.masked_where(self.mappable_bases.mask, agg_cutcounts)
        self.gp.logger.debug(f'Looking for outlier threshold {self.chrom_name}')
        outliers_tr = self.find_outliers_tr(agg_cutcounts_masked)

        high_signal_mask = (agg_cutcounts_masked >= outliers_tr).filled(False)

        self.gp.logger.debug(f'Fit model {self.chrom_name}')
        sliding_mean, sliding_variance = self.fit_model(agg_cutcounts_masked, high_signal_mask)
        r0 = (sliding_mean * sliding_mean) / (sliding_variance - sliding_mean)
        p0 = (sliding_variance - sliding_mean) / (sliding_variance)
        self.gp.logger.debug(f'Calculate p-value {self.chrom_name}')
        log_pvals = negbin_neglog10pvalue(agg_cutcounts_masked, r0, p0)
        data_df = pd.DataFrame({
            'log10_pval': log_pvals.filled(np.nan),
            'sliding_mean': sliding_mean.filled(np.nan),
            'sliding_variance': sliding_variance.filled(np.nan),
        }).reset_index(names='start')
        data_df['#chr'] = self.chrom_name

        self.gp.logger.debug(f"Chromosome {self.chrom_name} initial fit done")

        m0, v0 = self.fit_model(agg_cutcounts, high_signal_mask, in_window=False)

        params_df = pd.DataFrame({
            'chrom': [self.chrom_name],
            'outliers_tr': [outliers_tr],
            'mean': [m0],
            'variance': [v0],
            'rmsea': [np.nan],
        })
        self.gp.logger.debug(f"Chromosome {self.chrom_name} initial fit finished")
        return PeakCallingData(self.chrom_name, data_df, params_df)

    def extract_cutcounts(self, cutcounts_file):
        with TabixExtractor(
            cutcounts_file, 
            columns=['#chr', 'start', 'end', 'id', 'cutcounts']
        ) as cutcounts_loader:
            cutcounts = np.zeros(self.chrom_size, dtype=self.int_dtype)
            data = cutcounts_loader[self.genomic_interval]
            cutcounts[data['start'] - self.genomic_interval.start] = data['cutcounts'].to_numpy()
            assert cutcounts.shape[0] == self.chrom_size, "Cutcounts file does not match chromosome sizes"
        return cutcounts

    def fit_model(self, agg_cutcounts, high_signal_mask, in_window=True):
        if in_window:
            bg_sum_mappable = self.smooth_counts(
                self.mappable_bases,
                self.gp.bg_window,
                position_skip_mask=high_signal_mask
            )
            bg_sum_mappable = np.ma.masked_less(bg_sum_mappable, self.gp.min_mappable_bg)
            
            bg_sum = self.smooth_counts(agg_cutcounts, self.gp.bg_window, position_skip_mask=high_signal_mask)
            bg_sum_sq = self.smooth_counts(
                agg_cutcounts * agg_cutcounts,
                self.gp.bg_window,
                position_skip_mask=high_signal_mask
            )
        else:
            bg_sum_mappable = np.sum(self.mappable_bases[~high_signal_mask].compressed())
            compressed_cutcounts = agg_cutcounts[~high_signal_mask]
            bg_sum = np.sum(compressed_cutcounts)
            bg_sum_sq = np.sum(compressed_cutcounts * compressed_cutcounts)

        sliding_mean = bg_sum / bg_sum_mappable
        sliding_variance = (bg_sum_sq - bg_sum * sliding_mean) / (bg_sum_mappable - 1)

        return sliding_mean, sliding_variance
    
    def find_outliers_tr(self, aggregated_cutcounts):
        return np.quantile(aggregated_cutcounts.compressed(), self.gp.signal_tr)

    def get_mappable_bases(self, mappable_file):
        if mappable_file is None:
            mappable = np.ones(self.chrom_size, dtype=bool)
        else:
            try:
                with TabixExtractor(mappable_file, columns=['#chr', 'start', 'end']) as mappable_loader:
                    mappable = np.zeros(self.chrom_size, dtype=bool)
                    for _, row in mappable_loader[self.genomic_interval].iterrows():
                        if row['end'] > self.genomic_interval.end:
                            raise ValueError(f"Mappable bases file does not match chromosome sizes! Check input parameters. {row['end']} > {self.genomic_interval.end}")
                        mappable[row['start'] - self.genomic_interval.start:row['end'] - self.genomic_interval.end] = 1
                    assert mappable.shape[0] == self.chrom_size, "Mappable bases file does not match chromosome sizes"
            except ValueError:
                raise NoContigPresentError
        return ma.masked_where(~mappable, mappable)
    
    def smooth_counts(self, signal, window, position_skip_mask=None):
        return nan_moving_sum(
            signal,
            window=window,
            dtype=self.int_dtype,
            position_skip_mask=position_skip_mask
        )
        


def negbin_neglog10pvalue(x, r, p):
    x = ma.asarray(x)
    r = ma.asarray(r)
    p = ma.asarray(p)
    assert r.shape == p.shape, "r and p should have the same shape"
    if len(r.shape) == 0:
        resulting_mask = x.mask
    else:
        resulting_mask = reduce(ma.mask_or, [x.mask, r.mask, p.mask])
        r = r[~resulting_mask]
        p = p[~resulting_mask]
    result = ma.masked_where(resulting_mask, np.zeros(x.shape))
    result[~resulting_mask] = -st.nbinom.logsf(x[~resulting_mask] - 1, r, 1 - p) / np.log(10)
    return result


def nan_moving_sum(masked_array, window, dtype=None, position_skip_mask=None):
    if not isinstance(masked_array, ma.MaskedArray):
        masked_array = ma.masked_invalid(masked_array)

    if dtype is None:
        dtype = masked_array.dtype
    else:
        if dtype != masked_array.dtype:
            masked_array = masked_array.astype(dtype)

    data = masked_array.filled(0)
    if position_skip_mask is not None:
        assert position_skip_mask.shape == data.shape, "position_skip_mask should have the same shape as data"
        data[position_skip_mask] = 0

    conv_arr = np.ones(window, dtype=dtype)
    result = convolve(data, conv_arr, mode='same')
    result = ma.array(result, mask=masked_array.mask)
    return result


def read_chrom_sizes(chrom_sizes):
    return pd.read_table(
        chrom_sizes,
        header=None,
        names=['chrom', 'size']
    ).set_index('chrom')['size'].to_dict()


def main(cutcounts, chrom_sizes, mappable_bases_file, cpus):
    genome_processor = GenomeProcessor(chrom_sizes, mappable_bases_file, cpus=cpus, chromosomes=['chr19', 'chr20', 'chr21'])
    root_logger.debug('Calling peaks')
    return genome_processor.call_peaks(cutcounts)


if __name__ == "__main__":
    root_logger.debug('Processing started')
    cutcounts = sys.argv[1]
    chrom_sizes = read_chrom_sizes(sys.argv[2])
    mappable_bases_file = sys.argv[3]
    cpus = int(sys.argv[4])
    result, params = main(cutcounts, chrom_sizes, mappable_bases_file, cpus)
    result.to_parquet(sys.argv[5])
    params.to_csv(sys.argv[5] + '.params', sep='\t', header=True)
