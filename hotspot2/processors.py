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
from stats import calc_neglog10fdr, negbin_neglog10pvalue, nan_moving_sum, hotspots_from_log10_fdr_vectorized, modwt_smooth, find_varwidth_peaks
from utils import ProcessorOutputData, NoContigPresentError, ensure_contig_exists, read_df_for_chrom, normalize_density, run_bam2_bed, is_iterable, to_parquet_high_compression, delete
import sys
from typing import Iterable
import tempfile
import shutil
import os

root_logger = logging.getLogger(__name__)

counts_dtype = np.float32


def set_logger_config(logger: logging.Logger, level: int):
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)


class GenomeProcessor:
    """
    Main class to run hotspot2-related functions and store parameters.

    Parameters:
        - chrom_sizes: Dictionary containing chromosome sizes.
        - mappable_bases_file: Path to the tabix-indexed file containing mappable bases or None.
        - tmp_dir: Temporary directory for intermediate files. Will use system default if None.

        - window: Window size for aggregating cutcounts.
        - min_mappable: Minimum number of mappable bases for a window to be considered.
        - bg_window: Window size for aggregating background cutcounts.
        - min_mappable_bg: Minimum number of mappable bases for a window to be considered in background.

        - density_step: Step size for extracting density.
        - density_bandwidth: Bandwidth for MODWT smoothing.

        - signal_tr: Quantile threshold for outlier detection for background distribution fit.
        - fdr_method: Method for FDR calculation. 'bh and 'by' are supported. 'bh' (default) is tested.
        - cpus: Number of CPUs to use. Won't use more than the number of chromosomes.

        - chromosomes: List of chromosomes to process or None. Used mostly for debugging. Will generate wrong FDR corrections (only for these chromosomes).
        - save_debug: Save debug information.
        - modwt_level: Level of MODWT decomposition. 7 is tested.
        - logger_level: Logging level.
    """
    def __init__(
            self, chrom_sizes,
            mappable_bases_file=None,
            tmp_dir=None,
            window=201, min_mappable=101,
            bg_window=50001, min_mappable_bg=10000,
            density_step=20, density_bandwidth=151,
            signal_tr=0.975,
            fdr_method='bh',
            cpus=1,
            chromosomes=None,
            save_debug=False,
            modwt_level=7,
            logger_level=logging.INFO
        ) -> None:
        self.logger = root_logger
        self.save_debug = save_debug
        self.logger_level = logger_level
        self.chrom_sizes = chrom_sizes
        if chromosomes is not None:
            self.chrom_sizes = {k: v for k, v in chrom_sizes.items() if k in chromosomes}
        self.mappable_bases_file = mappable_bases_file
        self.tmp_dir = tmp_dir
        
        self.bg_window = bg_window
        self.window = window

        self.min_mappable = min_mappable
        self.min_mappable_bg = min_mappable_bg

        self.density_step = density_step
        self.density_bandwidth = density_bandwidth

        self.cpus = min(cpus, max(1, mp.cpu_count()))
        self.signal_tr = signal_tr
        self.fdr_method = fdr_method
        self.modwt_level = modwt_level

        self.chromosome_processors = sorted(
            [ChromosomeProcessor(self, chrom_name) for chrom_name in self.chrom_sizes.keys()],
            key=lambda x: x.chrom_size,
            reverse=True
        )
    
    def __getstate__(self): # Exclude logger and chromosome_processors from pickling
        state = self.__dict__
        if 'logger' in state:
            del state['logger']
        return state

    def set_logger(self):
        if not hasattr(self, 'logger'):
            self.logger = root_logger
            set_logger_config(self.logger, self.logger_level)

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)
        self.set_logger()

    def construct_parallel_args(self, *args):
        res_args = []
        for arg in args:
            if is_iterable(arg):
                if all(isinstance(x, ProcessorOutputData) for x in arg):
                     # if arg consits of ProcessorOutputData - 
                     # sort by chromosome name to match chromosome_processors
                    tmp = {x.identificator: x for x in arg}
                    reformat_arg = []
                    for x in self.chromosome_processors:
                        if x.chrom_name in tmp:
                            d = tmp[x.chrom_name]
                        else:
                            self.logger.debug(f"Chromosome {x.chrom_name} not found in input data. Skipping.")
                            d = None
                        reformat_arg.append(d)

                else:
                    assert len(arg) == len(self.chromosome_processors), f"Length of arguments must be equal to the number of chromosomes ({len(self.chromosome_processors)})."
                    reformat_arg = arg
            else:
                reformat_arg = [arg] * len(self.chromosome_processors)
            res_args.append(reformat_arg)
        return [self.chromosome_processors, *res_args]


    def parallel_by_chromosome(self, func, *args, cpus=None):
        """
        Basic function that handles parallel execution of a function by chromosome.
        """
        if cpus is None: # override cpus if provided
            cpus = self.cpus
        args = self.construct_parallel_args(*args)
        self.logger.debug(f'Using {cpus} CPUs for {func.__name__}')
        results = []
        if self.cpus == 1:
            for func_args in zip(*args):
                result = func(*func_args)
                if result is not None:
                    results.append(result)
        else:
            with ProcessPoolExecutor(max_workers=self.cpus) as executor:
                try:
                    for result in executor.map(func, *args):
                        if result is not None:
                            results.append(result)
                except Exception as e:
                    self.set_logger()
                    self.logger.critical("Exception, gracefully shutting down executor...")
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise e
        self.set_logger() # Restore logger after parallel execution
        self.logger.debug(f'Results of {func.__name__} emitted.')
        return results


    def calc_pval(self, cutcounts_file, fdrs_path, write_smoothing_params=False):
        self.logger.info('Calculating p-values')
        pvals_path = fdrs_path.replace('.fdrs', '.pvals')
        delete(pvals_path)
        params_outpath = pvals_path.replace('.pvals', '.pvals.params')
        delete(params_outpath)
        self.parallel_by_chromosome(
            ChromosomeProcessor.calc_pvals,
            cutcounts_file,
            pvals_path,
            params_outpath,
            write_smoothing_params
        )
        log10_pval = pd.read_parquet(
            pvals_path,
            engine='pyarrow', 
            columns=['chrom', 'log10_pval']
        ) 
        # file is always sorted within chromosomes
        chrom_pos_mapping = log10_pval['chrom'].drop_duplicates()
        starts = chrom_pos_mapping.index
        ends = [*starts[1:], log10_pval.shape[0]]
        log10_pval = log10_pval['log10_pval'].values
       
        self.logger.info('Calculating FDRs')
        fdrs = calc_neglog10fdr(log10_pval,fdr_method=self.fdr_method)
        del log10_pval
        gc.collect()

        fdrs = [
            ProcessorOutputData(
                chrom, 
                pd.DataFrame({'log10_fdr': fdrs[start:end]})
            )
            for chrom, start, end
            in zip(chrom_pos_mapping, starts, ends)
        ]
        gc.collect()
        delete(fdrs_path)
        self.logger.debug('Saving per-bp FDRs')
        self.parallel_by_chromosome(
            ChromosomeProcessor.to_parquet,
            fdrs,
            fdrs_path
        )
        return fdrs_path

    
    def call_hotspots(self, fdr_path, prefix, fdr_tr=0.05, min_width=50) -> ProcessorOutputData:
        """
        Call hotspots from path to parquet file containing log10(FDR) values.

        Parameters:
            - fdr_path: Path to the parquet file containing the log10(FDR) values.
            - fdr_tr: FDR threshold for calling hotspots.
            - min_width: Minimum width for a region to be called a hotspot.

        Returns:
            - hotspots: ProcessorOutputData containing the hotspots in bed format.
        """
        hotspots = self.parallel_by_chromosome(
            ChromosomeProcessor.call_hotspots,
            fdr_path,
            fdr_tr,
            min_width
        )
        hotspots = self.merge_and_add_chromosome(hotspots)
        hotspots.data_df['id'] = prefix
        hotspots.data_df['score'] = np.round(hotspots.data_df['max_neglog10_fdr'] * 10).astype(np.int64).clip(0, 1000)
        if len(hotspots.data_df) == 0:
            self.logger.critical(f"No hotspots called at FDR={fdr_tr}. Most likely something went wrong!")
        else:
            self.logger.debug(f"There are {len(hotspots.data_df)} hotspots called at FDR={fdr_tr}")
        return hotspots


    def modwt_smooth_signal(self, cutcounts_path, save_path):
        self.logger.info('Smoothing signal using MODWT')
        total_cutcounts = sum(
            self.parallel_by_chromosome(
                ChromosomeProcessor.total_cutcounts,
                cutcounts_path
            )
        )
        self.logger.debug('Total cutcounts = %d', total_cutcounts)
        
        delete(save_path)
        self.parallel_by_chromosome(
            ChromosomeProcessor.modwt_smooth_density,
            cutcounts_path,
            total_cutcounts,
            save_path
        )

    def write_cutcounts(self, bam_path, outpath) -> None:
        self.logger.info('Extracting cutcounts from bam file')
        run_bam2_bed(bam_path, outpath)

    def extract_density(self, smoothed_signal) -> ProcessorOutputData:
        data_df = []
        for chrom_processor in self.chromosome_processors:
            df = read_df_for_chrom(smoothed_signal, chrom_processor.chrom_name, columns=['chrom', 'normalized_density']).iloc[::self.density_step]
            if df.empty:
                continue
            df['start'] = np.arange(len(df)) * self.density_step
            data_df.append(df)
        data_df = pd.concat(data_df, ignore_index=True)
        data_df['end'] = data_df['start'] + self.density_step
        data_df['id'] = 'id'
        return ProcessorOutputData('all', data_df)


    def call_variable_width_peaks(self, smoothed_signal_path, hotspots_path) -> ProcessorOutputData:
        """
        Call variable width peaks from smoothed signal and hotspots.

        Parameters:
            - smoothed_signal: ProcessorOutputData containing the smoothed signal. (Output of GenomeProcessor.modwt_smooth_signal)
            - hotspots_path: Path to the tabix hotspots file.
        
        Returns:
            - peaks_data: ProcessorOutputData containing the peaks in bed format

        """
        peaks_data = self.parallel_by_chromosome(
            ChromosomeProcessor.call_variable_width_peaks,
            smoothed_signal_path,
            hotspots_path
        )
        return self.merge_and_add_chromosome(peaks_data)

    def merge_and_add_chromosome(self, results: Iterable[ProcessorOutputData]) -> ProcessorOutputData:
        data = []
        results = sorted(results, key=lambda x: x.identificator)
        categories = [x.identificator for x in results]
        for res in results:
            df = res.data_df
            df['chrom'] = pd.Categorical(
                [res.identificator] * df.shape[0],
                categories=categories,
            )
            data.append(df)
            
        data = pd.concat(data, ignore_index=True)
        return ProcessorOutputData('all', data)



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
        self.mappable_bases = None
    
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
    
    def extract_cutcounts(self, cutcounts_file):
        cutcounts = np.zeros(self.chrom_size, dtype=counts_dtype)
        self.extract_mappable_bases()
        try:
            self.gp.logger.debug(f'Extracting cutcounts for chromosome {self.chrom_name}')
            with TabixExtractor(cutcounts_file) as cutcounts_loader:
                data = cutcounts_loader[self.genomic_interval]
                assert data.eval('end - start == 1').all(), "Cutcounts are expected to be at basepair resolution"
                cutcounts[data['start']] = data['count'].to_numpy()
        except ValueError:
            raise NoContigPresentError
        return cutcounts


    @ensure_contig_exists
    def calc_pvals(self, cutcounts_file, pvals_outpath, params_outpath, write_raw_pvals=False) -> ProcessorOutputData:
        cutcounts = self.extract_cutcounts(cutcounts_file)

        self.gp.logger.debug(f'Aggregating cutcounts for chromosome {self.chrom_name}')
        agg_cutcounts = self.smooth_counts(cutcounts, self.gp.window)
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
        data = {'log10_pval': log_pvals.filled(np.nan).astype(np.float16)}
        if self.gp.save_debug or write_raw_pvals:
            data.update({
                'sliding_mean': sliding_mean.filled(np.nan).astype(np.float16),
                'sliding_variance': sliding_variance.filled(np.nan).astype(np.float16),
            })
        del r0, p0
        gc.collect()
        data = pd.DataFrame.from_dict(data)
        vals, counts = np.unique(agg_cutcounts[~high_signal_mask].compressed(), return_counts=True)
        params_df = pd.DataFrame({
            'cutcounts': vals,
            'count': counts,
            'outliers_tr': [outliers_tr] * len(vals),
            'mean': [m0] * len(vals),
            'variance': [v0] * len(vals),
        })
        self.gp.logger.debug(f"Writing pvals for {self.chrom_name}")
        self.to_parquet(data, pvals_outpath)
        self.to_parquet(params_df, params_outpath)


    @ensure_contig_exists
    def call_hotspots(self, fdr_path, fdr_threshold=0.05, min_width=50) -> ProcessorOutputData:
        columns = ['log10_fdr']
        log10_fdr_array = read_df_for_chrom(fdr_path, self.chrom_name, columns=columns)['log10_fdr'].to_numpy()
        if log10_fdr_array.size == 0:
            raise NoContigPresentError
        self.gp.logger.debug(f"Calling hotspots for {self.chrom_name}")
        data = hotspots_from_log10_fdr_vectorized(
            log10_fdr_array,
            fdr_threshold=fdr_threshold,
            min_width=min_width
        )
        return ProcessorOutputData(self.chrom_name, data) if data is not None else None
    

    @ensure_contig_exists
    def modwt_smooth_density(self, cutcounts_path, total_cutcounts, save_path) -> ProcessorOutputData:
        """
        Run MODWT smoothing on cutcounts.
        """
        cutcounts = self.extract_cutcounts(cutcounts_path)
        agg_counts = self.smooth_counts(cutcounts, self.gp.density_bandwidth).filled(0)
        filters = 'haar'
        level = self.gp.modwt_level
        self.gp.logger.debug(f"Running modwt smoothing (filter={filters}, level={level}) for {self.chrom_name}")
        smoothed = modwt_smooth(agg_counts, filters, level=level)
        data = pd.DataFrame({
            #'cutcounts': cutcounts, 
            'smoothed': smoothed,
            'normalized_density': normalize_density(agg_counts, total_cutcounts) 
            # maybe use the same window as for pvals? then cutcounts is redundant
        })
        self.to_parquet(data, save_path)
        return ProcessorOutputData(self.chrom_name, save_path)


    @ensure_contig_exists
    def call_variable_width_peaks(self, smoothed_signal_path, hotspots_path) -> ProcessorOutputData:
        signal_df = read_df_for_chrom(smoothed_signal_path, self.chrom_name, columns=['smoothed', 'normalized_density'])
        if signal_df.empty:
            raise NoContigPresentError

        hotspots_coordinates = self.read_hotspots_tabix(hotspots_path)
        hotspot_starts = hotspots_coordinates['start'].values
        hotspot_ends = hotspots_coordinates['end'].values

        normalized_density = signal_df['normalized_density'].values

        peaks_in_hotspots_trimmed, _ = find_varwidth_peaks(
            signal_df['smoothed'].values,
            hotspot_starts,
            hotspot_ends
        )
        peaks_df = pd.DataFrame(
            peaks_in_hotspots_trimmed,
            columns=['start', 'summit', 'end']
        )
        peaks_df['summit_density'] = normalized_density[peaks_df['summit']]
        
        peaks_df['max_density'] = [
            np.max(normalized_density[start:end])
            for start, end in zip(peaks_df['start'], peaks_df['end'])]

        return ProcessorOutputData(self.chrom_name, peaks_df)


    def read_hotspots_tabix(self, hotspots):
        with TabixExtractor(hotspots) as hotspots_loader:
            df = hotspots_loader[self.genomic_interval]
        return df

    def fit_background_negbin_model(self, agg_cutcounts, high_signal_mask, in_window=True):
        if in_window:
            bg_sum_mappable = self.smooth_counts(
                self.mappable_bases,
                self.gp.bg_window,
                position_skip_mask=high_signal_mask
            )
            bg_sum_mappable = np.ma.masked_less(bg_sum_mappable, self.gp.min_mappable_bg)
            self.gp.logger.debug(f"Background mappable bases calculated for {self.chrom_name}")
            
            bg_sum = self.smooth_counts(
                agg_cutcounts,
                self.gp.bg_window,
                position_skip_mask=high_signal_mask
            )
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

        sliding_mean = (bg_sum / bg_sum_mappable)
        sliding_variance = ((bg_sum_sq - bg_sum * sliding_mean) / (bg_sum_mappable - 1))

        return sliding_mean, sliding_variance
        
    
    def smooth_counts(self, signal: np.ndarray, window: int, dtype=None, position_skip_mask: np.ndarray=None):
        """
        Wrapper for nan_moving_sum to smooth cutcounts.
        Might need to remove the method and call nan_moving_sum directly.
        """
        return nan_moving_sum(
            signal,
            window=window,
            dtype=dtype,
            position_skip_mask=position_skip_mask
        )
    
    @ensure_contig_exists
    def to_parquet(self, data_df, path):
        """
        Workaround for writing parquet files for chromosomes in parallel.
        """
        if data_df is None:
            raise NoContigPresentError
        if isinstance(data_df, ProcessorOutputData):
            data_df = data_df.data_df
        data_df['chrom'] = pd.Categorical(
            [self.chrom_name] * data_df.shape[0],
            categories=[x for x in self.gp.chrom_sizes.keys()]
        )
        os.makedirs(path, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.gp.tmp_dir) as temp_dir:
            temp_path = os.path.join(temp_dir, f'{self.chrom_name}.temp.parquet')
            to_parquet_high_compression(data_df, temp_path)
            shutil.move(os.path.join(temp_path, f'chrom={self.chrom_name}'), path)
        
    
    @ensure_contig_exists
    def total_cutcounts(self, cutcounts):
        self.gp.logger.debug(f"Calculating total cutcounts for {self.chrom_name}")
        return self.extract_cutcounts(cutcounts).sum()