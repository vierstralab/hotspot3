import logging
import numpy as np
import numpy.ma as ma
from scipy.signal import convolve
import scipy.stats as st
from functools import reduce
from concurrent.futures import ProcessPoolExecutor
from genome_tools.genomic_interval import GenomicInterval
from genome_tools.data.extractors import TabixExtractor
import multiprocessing as mp
import pandas as pd
import dataclasses
from statsmodels.stats.multitest import multipletests
from typing import Tuple
import sys
import gc


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
    def __init__(self, chrom_sizes, mappable_bases_file=None, window=201, bg_window=50001, min_mappable_bg=10000, signal_tr=0.975, int_dtype = np.int32, fdr_method='fdr_bh', cpus=1, chromosomes=None, save_debug=False, logger_level=logging.DEBUG) -> None:
        self.logger = root_logger
        self.save_debug = save_debug
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
        self.signal_tr = signal_tr
        self.fdr_method = fdr_method

        self.chromosome_processors = [x for x in self.get_chromosome_processors()]
        self.logger.info(f"Chromosomes with mappable track: {len(self.chromosome_processors)}")
    
    def __getstate__(self):
        state = self.__dict__
        if 'logger' in state:
            del state['logger']
        return state

    def restore_logger(self):
        if not hasattr(self, 'logger'):
            self.logger = root_logger
            set_logger_config(self.logger, self.logger_level)

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)
        self.restore_logger()

    def get_chromosome_processors(self):
        for chrom_name in self.chrom_sizes.keys():
            try:
                yield ChromosomeProcessor(self, chrom_name)
            except NoContigPresentError:
                continue
        
    def call_hotspots(self, cutcounts_file) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sorted_processors = sorted(
            self.chromosome_processors,
            key=lambda x: x.chrom_size,
            reverse=True
        )
        if self.cpus == 1:
            self.logger.debug('Using 1 CPU')
            result = []
            for chrom_processor in sorted_processors:
                res = chrom_processor.calc_pvals(cutcounts_file)
                result.append(res)
    
        else:
            self.logger.debug(f'Using {self.cpus} CPUs')
            with ProcessPoolExecutor(max_workers=self.cpus) as executor:
                results = executor.map(
                    ChromosomeProcessor.calc_pvals, sorted_processors, [cutcounts_file] * len(sorted_processors)
                )

        self.restore_logger()
        self.logger.debug('Concatenating results')

        data_df, params_df = self.merge_dfs(results)
        self.logger.debug('Results concatenated. Calculating FDR')
        
        data_df['log10_fdr'] = self.calc_fdr(data_df['log10_pval'])

        result_columns = ['#chr', 'start', 'log10_fdr']
        if self.save_debug:
            result_columns += ['log10_pval', 'sliding_mean', 'sliding_variance']
        data_df = data_df[result_columns]
        return data_df, params_df
    
    def merge_dfs(self, results: list[PeakCallingData]) -> pd.DataFrame:
        data = []
        params = []
        for res in results:
            df = res.data_df
            df['#chr'] = np.asarray([res.chrom] * df.shape[0], dtype='S10')
            df['start'] = np.arange(0, df.shape[0], dtype=np.int32)
            params.append(res.params_df)
            data.append(df)
        return pd.concat(data), pd.concat(params)

    def calc_fdr(self, pval_list):
        log_fdr = np.empty(pval_list.shape)
        not_nan = ~np.isnan(pval_list)
        log_fdr[~not_nan] = np.nan
        log_fdr[not_nan] = -np.log10(multipletests(np.power(10, -pval_list[not_nan]), method=self.fdr_method)[1])
        return log_fdr.astype(np.float32)


class ChromosomeProcessor:
    def __init__(self, genome_processor: GenomeProcessor, chrom_name: str) -> None:
        self.chrom_name = chrom_name
        self.gp = genome_processor
        self.chrom_size = self.gp.chrom_sizes[chrom_name]
        self.genomic_interval = GenomicInterval(chrom_name, 0, self.chrom_size)
  
        self.int_dtype = self.gp.int_dtype
        self.mappable_bases = self.get_mappable_bases()

    def calc_pvals(self, cutcounts_file) -> PeakCallingData:
        self.gp.logger.debug(f'Extracting cutcounts for chromosome {self.chrom_name}')
        cutcounts = self.extract_cutcounts(cutcounts_file)

        self.gp.logger.debug(f'Aggregating cutcounts for chromosome {self.chrom_name}')
        agg_cutcounts = self.smooth_counts(cutcounts, self.gp.window)
        agg_cutcounts = np.ma.masked_where(self.mappable_bases.mask, agg_cutcounts)
        self.gp.logger.debug(
            f"Cutcounts aggregated for {self.chrom_name}, {agg_cutcounts.count()}/{agg_cutcounts.shape[0]} bases are mappable")

        del cutcounts
        gc.collect()

        outliers_tr = self.find_outliers_tr(agg_cutcounts)
        self.gp.logger.debug(f'Found outlier threshold={outliers_tr:1f} for {self.chrom_name}')

        high_signal_mask = (agg_cutcounts >= outliers_tr).filled(False)
        self.gp.logger.debug(f'Fit model for {self.chrom_name}')

        m0, v0 = self.fit_model(agg_cutcounts, high_signal_mask, in_window=False)
        self.gp.logger.debug(f"Total fit finished for {self.chrom_name}")

        sliding_mean, sliding_variance = self.fit_model(agg_cutcounts, high_signal_mask)

        r0 = (sliding_mean * sliding_mean) / (sliding_variance - sliding_mean)
        p0 = (sliding_variance - sliding_mean) / (sliding_variance)
        if not self.gp.save_debug:
            del sliding_mean, sliding_variance
            gc.collect()

        self.gp.logger.debug(f'Calculate p-value for {self.chrom_name}')
        log_pvals = negbin_neglog10pvalue(agg_cutcounts, r0, p0)

        self.gp.logger.debug(f"Window fit finished for {self.chrom_name}")
        data = {'log10_pval': log_pvals.filled(np.nan)}
        if self.gp.save_debug:
            data.update({
                'sliding_mean': sliding_mean.filled(np.nan),
                'sliding_variance': sliding_variance.filled(np.nan),
            })
        else:
            del r0, p0
            gc.collect()
        data_df = pd.DataFrame.from_dict(data)

        params_df = pd.DataFrame({
            'chrom': [self.chrom_name],
            'outliers_tr': [outliers_tr],
            'mean': [m0],
            'variance': [v0],
            'rmsea': [np.nan],
        })
        return PeakCallingData(self.chrom_name, data_df, params_df)

    def extract_cutcounts(self, cutcounts_file):
        with TabixExtractor(
            cutcounts_file, 
            columns=['#chr', 'start', 'end', 'id', 'cutcounts']
        ) as cutcounts_loader:
            cutcounts = np.zeros(self.chrom_size, dtype=self.int_dtype)
            data = cutcounts_loader[self.genomic_interval]
            cutcounts[data['start'] - self.genomic_interval.start] = data['cutcounts'].to_numpy()
        return cutcounts

    def fit_model(self, agg_cutcounts, high_signal_mask, in_window=True):
        if in_window:
            bg_sum_mappable = self.smooth_counts(
                self.mappable_bases,
                self.gp.bg_window,
                position_skip_mask=high_signal_mask
            )
            bg_sum_mappable = np.ma.masked_less(bg_sum_mappable, self.gp.min_mappable_bg)
            self.gp.logger.debug(f"Background mappable bases calculated for {self.chrom_name}")
            bg_sum = self.smooth_counts(agg_cutcounts, self.gp.bg_window, position_skip_mask=high_signal_mask)
            bg_sum_sq = self.smooth_counts(
                agg_cutcounts * agg_cutcounts,
                self.gp.bg_window,
                position_skip_mask=high_signal_mask
            )
            self.gp.logger.debug(f"Background cutcounts calculated for {self.chrom_name}")
        else:
            bg_sum_mappable = np.sum(self.mappable_bases[~high_signal_mask].compressed())
            compressed_cutcounts = agg_cutcounts[~high_signal_mask]
            bg_sum = np.sum(compressed_cutcounts)
            bg_sum_sq = np.sum(compressed_cutcounts * compressed_cutcounts)

        sliding_mean = (bg_sum / bg_sum_mappable).astype(np.float32)
        sliding_variance = ((bg_sum_sq - bg_sum * sliding_mean) / (bg_sum_mappable - 1)).astype(np.float32)

        return sliding_mean, sliding_variance
    
    def find_outliers_tr(self, aggregated_cutcounts):
        return np.quantile(aggregated_cutcounts.compressed(), self.gp.signal_tr)

    def get_mappable_bases(self):
        mappable_file = self.gp.mappable_bases_file
        if mappable_file is None:
            mappable = np.ones(self.chrom_size, dtype=bool)
        else:
            try:
                with TabixExtractor(mappable_file, columns=['#chr', 'start', 'end']) as mappable_loader:
                    mappable = np.zeros(self.chrom_size, dtype=bool)
                    for _, row in mappable_loader[self.genomic_interval].iterrows():
                        if row['end'] > self.genomic_interval.end:
                            raise ValueError(f"Mappable bases file does not match chromosome sizes! Check input parameters. {row['end']} > {self.genomic_interval.end} for {self.chrom_name}")
                        mappable[row['start'] - self.genomic_interval.start:row['end'] - self.genomic_interval.end] = 1
            except ValueError:
                raise NoContigPresentError
        self.gp.logger.debug(f"Chromosome {self.chrom_name} mappable bases extracted. {np.sum(mappable)}/{self.chrom_size} are mappable")
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
    result = ma.masked_where(resulting_mask, np.zeros(x.shape, dtype=np.float32))
    result[~resulting_mask] = -st.nbinom.logsf(x[~resulting_mask] - 1, r, 1 - p) / np.log(10)
    return result.astype(np.float32)


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


def merge_regions_log10_fdr_vectorized(log10_fdr_array, threshold=0.05, min_width=50):
    """
    Merge adjacent base pairs in a NumPy array where log10(FDR) is below the threshold.

    Parameters:
    - log10_fdr_array: NumPy array of log10(FDR) values, length of the chromosome.
    - threshold: The log10(FDR) threshold for merging.
    - min_width: Minimum width for a merged region.

    Returns:
    - start_indices: Array of start positions of the merged regions.
    - end_indices: Array of end positions of the merged regions.
    - min_log10_fdr_values: Array of minimum log10(FDR) values within each region.
    """
    below_threshold = log10_fdr_array >= -np.log10(threshold)
    # Diff returns -1 for transitions from True to False, 1 for transitions from False to True
    boundaries = np.diff(below_threshold.astype(np.int8), prepend=0, append=0).astype(np.int8)

    region_starts = np.where(boundaries == 1)[0]
    region_ends = np.where(boundaries == -1)[0]

    # Filter regions by minimum width
    valid_widths = (region_ends - region_starts) >= min_width
    region_starts = region_starts[valid_widths]
    region_ends = region_ends[valid_widths]

    min_log10_fdr_values = np.empty(region_ends.shape)  # Pre-allocate array
    for i in range(len(region_starts)):
        start = region_starts[i]
        end = region_ends[i]
        min_log10_fdr_values[i] = np.min(log10_fdr_array[start:end])

    return region_starts, region_ends, min_log10_fdr_values


def main(cutcounts, chrom_sizes, mappable_bases_file, cpus):
    genome_processor = GenomeProcessor(
        chrom_sizes,
        mappable_bases_file,
        cpus=cpus,
        #chromosomes=['chr1', 'chr2', 'chr20', 'chr19']
    )
    root_logger.debug('Calling hotspots')
    return genome_processor.call_hotspots(cutcounts)


if __name__ == "__main__":
    set_logger_config(root_logger, logging.DEBUG)
    root_logger.debug('Processing started')
    cutcounts = sys.argv[1]
    chrom_sizes = read_chrom_sizes(sys.argv[2])
    mappable_bases_file = sys.argv[3]
    cpus = int(sys.argv[4])
    result, params = main(cutcounts, chrom_sizes, mappable_bases_file, cpus)
    root_logger.debug('Saving results')
    result.to_parquet(sys.argv[5], compression='zstd')
    params.to_csv(sys.argv[5] + '.params.gz', sep='\t', header=True)
