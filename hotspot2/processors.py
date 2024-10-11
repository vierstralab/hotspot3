import logging
import numpy as np
import numpy.ma as ma
from concurrent.futures import ProcessPoolExecutor
from genome_tools.genomic_interval import GenomicInterval
from genome_tools.data.extractors import TabixExtractor
import multiprocessing as mp
import pandas as pd
import sys
import gc
from stats import calc_log10fdr, negbin_neglog10pvalue, nan_moving_sum, hotspots_from_log10_fdr_vectorized
from utils import arg_to_list, ProcessorOutputData, merge_and_add_chromosome


root_logger = logging.getLogger(__name__)


def set_logger_config(logger: logging.Logger, level: int):
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)


class NoContigPresentError(Exception):
    ...


class GenomeProcessor:
    """
    Main class to run hotspot2-related functions and store parameters.
    """
    def __init__(
            self, chrom_sizes,
            mappable_bases_file=None,
            window=201, min_mappable=101,
            bg_window=50001, min_mappable_bg=10000,
            density_step=20, density_bandwidth=151,
            signal_tr=0.975,
            int_dtype = np.int32,
            fdr_method='fdr_bh',
            cpus=1,
            chromosomes=None,
            save_debug=False,
            logger_level=logging.INFO
        ) -> None:
        self.logger = root_logger
        self.save_debug = save_debug
        self.logger_level = logger_level
        self.chrom_sizes = chrom_sizes
        if chromosomes is not None:
            self.chrom_sizes = {k: v for k, v in chrom_sizes.items() if k in chromosomes}
        self.mappable_bases_file = mappable_bases_file
        
        self.bg_window = bg_window
        self.window = window

        self.min_mappable = min_mappable
        self.min_mappable_bg = min_mappable_bg

        self.density_step = density_step
        self.density_bandwidth = density_bandwidth

        self.int_dtype = int_dtype
        self.cpus = min(cpus, max(1, mp.cpu_count()))
        self.signal_tr = signal_tr
        self.fdr_method = fdr_method

        self.chromosome_processors = sorted(
            [ChromosomeProcessor(self, chrom_name) for chrom_name in self.chrom_sizes.keys()],
            key=lambda x: x.chrom_size,
            reverse=True
        )
    
    def __getstate__(self): # Exclude logger and chromosome_processors from pickling
        state = self.__dict__
        if 'logger' in state:
            del state['logger']
        if 'chromosome_processors' in state:
            del state['chromosome_processors']
        return state

    def set_logger(self):
        if not hasattr(self, 'logger'):
            self.logger = root_logger
            set_logger_config(self.logger, self.logger_level)

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)
        self.set_logger()

    def parallel_by_chromosome(self, func, *args) -> ProcessorOutputData:
        all_args = zip(self.chromosome_processors, *[arg_to_list(arg) for arg in args])
        if self.cpus == 1:
            results = [func(*func_args) for func_args in all_args]
        else:
            with ProcessPoolExecutor(max_workers=self.cpus) as executor:
                results = list(executor.map(func, *all_args))
        self.set_logger() # Restore logger after parallel execution
        for res in results:
            if res is None:
                self.logger.debug(f"{res.chrom_name} not found in {args[0]}. Skipping...")
        return merge_and_add_chromosome([x for x in results if x is not None])


    def calc_pval(self, cutcounts_file, write_raw_pvals=False) -> ProcessorOutputData:
        self.logger.debug(f'Using {self.cpus} CPUs')
        merged_data = self.parallel_by_chromosome(ChromosomeProcessor.calc_pvals, cutcounts_file)
    
        self.logger.debug('Results concatenated. Calculating FDR')

        merged_data.data_df['log10_fdr'] = calc_log10fdr(
            merged_data.data_df['log10_pval'],
            fdr_method=self.fdr_method
        )

        result_columns = ['chrom', 'log10_fdr']
        if self.save_debug or write_raw_pvals:
            result_columns += ['log10_pval', 'sliding_mean', 'sliding_variance']
        merged_data.data_df = merged_data.data_df[result_columns] 
    
        return merged_data


    def calc_density(self, cutcounts_file) -> ProcessorOutputData:
        merged_data = self.parallel_by_chromosome(ChromosomeProcessor.extract_cutcounts, cutcounts_file)
        return merged_data
    

    def call_hotspots(self, fdr_path, fdr_tr=0.05, min_width=50) -> ProcessorOutputData:
        """
        Call hotspots from path to parquet file containing log10(FDR) values.

        Parameters:
            - fdr_path: Path to the parquet file containing the log10(FDR) values.
            - fdr_tr: FDR threshold for calling hotspots.
            - min_width: Minimum width for a region to be called a hotspot.

        Returns:
            - hotspots: BED-like DataFrame containing hotspots.
        """
        hotspots = self.parallel_by_chromosome(
            ChromosomeProcessor.call_hotspots,
            fdr_path,
            fdr_tr,
            min_width
        )
        return pd.concat(hotspots, ignore_index=True)


class ChromosomeProcessor:
    """
    Individual chromosome processor. Used for parallel processing of chromosomes.
    Don't use directly, use GenomeProcessor instead.
    """
    def __init__(self, genome_processor: GenomeProcessor, chrom_name: str) -> None:
        self.chrom_name = chrom_name
        self.gp = genome_processor
        self.chrom_size = self.gp.chrom_sizes[chrom_name]
        self.genomic_interval = GenomicInterval(chrom_name, 0, self.chrom_size)
        self.int_dtype = self.gp.int_dtype
        self.mappable_bases = None
        self.cutcounts = None
        self.last_cutcounts_file = None
    
    def extract_mappable_bases(self, force=False):
        if self.mappable_bases is not None and not force:
            return
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

        self.mappable_bases = ma.masked_where(~mappable, mappable)
        self.gp.logger.debug(f"Chromosome {self.chrom_name} mappable bases extracted. {np.sum(mappable)}/{self.chrom_size} are mappable")
    
    def extract_cutcounts(self, cutcounts_file, force=False):
        if self.cutcounts is not None and not force and cutcounts_file == self.last_cutcounts_file:
            return
        self.gp.logger.debug(f'Extracting cutcounts for chromosome {self.chrom_name}')
        cutcounts = np.zeros(self.chrom_size, dtype=self.int_dtype)
        try:
            with TabixExtractor(
                cutcounts_file, columns=['#chr', 'start', 'end', 'id', 'cutcounts']
            ) as cutcounts_loader:
                
                data = cutcounts_loader[self.genomic_interval]
                cutcounts[data['start']] = data['cutcounts'].to_numpy()
        except ValueError:
            raise NoContigPresentError
        self.cutcounts = cutcounts
        self.last_cutcounts_file = cutcounts_file

    def calc_density(self, cutcounts_file) -> ProcessorOutputData:
        try:
            self.extract_cutcounts(cutcounts_file)
        except NoContigPresentError: # FIXME handle in decorator
            return
        self.gp.logger.debug(f'Calculating density {self.chrom_name}')
        density = self.smooth_counts(
            self.cutcounts,
            self.gp.density_bandwidth
        )[::self.gp.density_step].filled(0)
        return ProcessorOutputData(self.chrom_name, pd.DataFrame({'density': density}))

    def calc_pvals(self, cutcounts_file) -> ProcessorOutputData:
        try:
            self.extract_mappable_bases()
            self.extract_cutcounts(cutcounts_file)
        except NoContigPresentError: # FIXME handle in decorator
            return

        self.gp.logger.debug(f'Aggregating cutcounts for chromosome {self.chrom_name}')
        agg_cutcounts = self.smooth_counts(self.cutcounts, self.gp.window)
        agg_cutcounts = np.ma.masked_where(self.mappable_bases.mask, agg_cutcounts)
        self.gp.logger.debug(
            f"Cutcounts aggregated for {self.chrom_name}, {agg_cutcounts.count()}/{agg_cutcounts.shape[0]} bases are mappable")

        outliers_tr = np.quantile(agg_cutcounts.compressed(), self.gp.signal_tr)
        self.gp.logger.debug(f'Found outlier threshold={outliers_tr:.1f} for {self.chrom_name}')

        high_signal_mask = (agg_cutcounts >= outliers_tr).filled(False)
        self.gp.logger.debug(f'Fit model for {self.chrom_name}')

        m0, v0 = self.fit_background_negbin_model(agg_cutcounts, high_signal_mask, in_window=False)
        self.gp.logger.debug(f"Total fit finished for {self.chrom_name}")

        sliding_mean, sliding_variance = self.fit_background_negbin_model(agg_cutcounts, high_signal_mask)

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
        return ProcessorOutputData(self.chrom_name, data_df, params_df)

    def call_hotspots(self, fdr_path, fdr_threshold=0.05, min_width=50) -> ProcessorOutputData:
        data = hotspots_from_log10_fdr_vectorized(self.chrom_name, fdr_path, fdr_threshold=fdr_threshold, min_width=min_width)
        return ProcessorOutputData(self.chrom_name, data) if data is not None else None
    
    def call_variable_width_peaks(self, density_path, hotspots):
        raise NotImplementedError

    def fit_background_negbin_model(self, agg_cutcounts, high_signal_mask, in_window=True):
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
        
    
    def smooth_counts(self, signal, window, position_skip_mask=None):
        return nan_moving_sum(
            signal,
            window=window,
            dtype=self.int_dtype,
            position_skip_mask=position_skip_mask
        )
    