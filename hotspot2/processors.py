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
from stats import calc_log10fdr, negbin_neglog10pvalue, nan_moving_sum, hotspots_from_log10_fdr_vectorized, modwt_smooth, find_varwidth_peaks
from utils import ProcessorOutputData, merge_and_add_chromosome,  NoContigPresentError, ensure_contig_exists, read_df_for_chrom, normalize_density, run_bam2_bed, is_iterable


root_logger = logging.getLogger(__name__)


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

        - window: Window size for aggregating cutcounts.
        - min_mappable: Minimum number of mappable bases for a window to be considered.
        - bg_window: Window size for aggregating background cutcounts.
        - min_mappable_bg: Minimum number of mappable bases for a window to be considered in background.

        - density_step: Step size for extracting density.
        - density_bandwidth: Bandwidth for MODWT smoothing.

        - signal_tr: Quantile threshold for outlier detection for background distribution fit.
        - int_dtype: Integer type for cutcounts. int32 (default) should be sufficient for most cases.
        - fdr_method: Method for FDR calculation. 'fdr_bh' (default) is tested.
        - cpus: Number of CPUs to use. Won't use more than the number of chromosomes.

        - chromosomes: List of chromosomes to process or None. Used mostly for debugging. Will generate wrong FDR corrections (only for these chromosomes).
        - save_debug: Save debug information.
        - modwt_level: Level of MODWT decomposition. 7 is tested.
        - logger_level: Logging level.
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

    def parallel_by_chromosome(self, func, *args, cpus=None) -> list[ProcessorOutputData]:
        if cpus is None: # override cpus if provided
            cpus = self.cpus
        all_args = self.construct_parallel_args(*args)
        self.logger.debug(f'Using {cpus} CPUs for {func.__name__}')
        if self.cpus == 1:
            results = [func(*func_args) for func_args in all_args]
        else:
            with ProcessPoolExecutor(max_workers=self.cpus) as executor:
                try:
                    results = list(executor.map(func, *all_args))
                except Exception as e:
                    self.set_logger()
                    self.logger.critical("Exception, gracefully shutting down executor...")
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise e
        self.set_logger() # Restore logger after parallel execution
        self.logger.debug(f'Results of {func.__name__} collected.')
        return [x for x in results if x is not None]


    def calc_pval(self, cutcounts_file, write_raw_pvals=False) -> ProcessorOutputData:
        merged_data = self.parallel_by_chromosome(ChromosomeProcessor.calc_pvals, cutcounts_file)
        merged_data = merge_and_add_chromosome(merged_data)
    
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
        hotspots = merge_and_add_chromosome(hotspots)
        hotspots.data_df['id'] = prefix
        hotspots.data_df['score'] = np.round(hotspots.data_df['max_neglog10_fdr'] * 10).astype(np.int64).clip(0, 1000)
        return hotspots


    def modwt_smooth_signal(self, cutcounts_path) -> list[ProcessorOutputData]:
        self.logger.info('Smoothing signal using MODWT')
        modwt_data = self.parallel_by_chromosome(
            ChromosomeProcessor.modwt_smooth_density,
            cutcounts_path,
        )
        gc.collect()
        total_cutcounts = np.sum([x.extra_df['total_cutcounts'].values for x in modwt_data])

        self.logger.info(f'Normalizing density with total cutcounts={total_cutcounts}')

        modwt_data = self.parallel_by_chromosome(
            ChromosomeProcessor.normalize_density,
            modwt_data,
            total_cutcounts,
            cpus=1
        )
        gc.collect()

        return modwt_data

    def write_cutcounts(self, bam_path, outpath) -> None:
        run_bam2_bed(bam_path, outpath)

    def extract_density(self, smoothed_signal: list[ProcessorOutputData]) -> ProcessorOutputData:
        # Optimization to avoid storing full pd.concat in memory
        for sig in smoothed_signal: # Take every density_step-th element
            sig.data_df = sig.data_df.iloc[np.arange(0, len(sig.data_df), self.density_step)]
            sig.data_df['start'] = np.arange(len(sig.data_df)) * self.density_step
        data_df = merge_and_add_chromosome(smoothed_signal).data_df
        data_df['end'] = data_df['start'] + self.density_step
        return ProcessorOutputData('all', data_df)


    def call_variable_width_peaks(self, smoothed_data: list[ProcessorOutputData], hotspots_path) -> ProcessorOutputData:
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
                smoothed_data,
                hotspots_path
            )
        return merge_and_add_chromosome(peaks_data)


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
        cutcounts = np.zeros(self.chrom_size, dtype=self.int_dtype)
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
    def calc_pvals(self, cutcounts_file) -> ProcessorOutputData:
        cutcounts = self.extract_cutcounts(cutcounts_file)

        self.gp.logger.debug(f'Aggregating cutcounts for chromosome {self.chrom_name}')
        agg_cutcounts = self.smooth_int_counts(cutcounts, self.gp.window)
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


    @ensure_contig_exists
    def call_hotspots(self, fdr_path, fdr_threshold=0.05, min_width=50) -> ProcessorOutputData:
        log10_fdr_array = read_df_for_chrom(fdr_path, self.chrom_name)['log10_fdr'].to_numpy()
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
    def modwt_smooth_density(self, cutcounts_path) -> ProcessorOutputData:
        """
        Run MODWT smoothing on cutcounts.
        """
        cutcounts = self.extract_cutcounts(cutcounts_path)
        agg_counts = self.smooth_int_counts(cutcounts, self.gp.density_bandwidth).filled(0)
        filters = 'haar'
        level = self.gp.modwt_level
        self.gp.logger.debug(f"Running modwt smoothing (filter={filters}, level={level}) for {self.chrom_name}")
        smoothed = modwt_smooth(agg_counts, filters, level=level).astype(np.float32)
        extra_df = pd.DataFrame({'total_cutcounts': [np.sum(cutcounts)]})
        data = pd.DataFrame({
            #'cutcounts': cutcounts, 
            'smoothed': smoothed,
            'density': agg_counts # maybe use the same window as for pvals? then cutcounts is redundant
        })
        return ProcessorOutputData(self.chrom_name, data, extra_df)


    @ensure_contig_exists
    def call_variable_width_peaks(self, signal_data: ProcessorOutputData, hotspots: str) -> ProcessorOutputData:
        if signal_data is None:
            raise NoContigPresentError
        signal_df = signal_data.data_df
        hotspots_coordinates = self.read_hotspots_tabix(hotspots)
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
            bg_sum_mappable = self.smooth_int_counts(
                self.mappable_bases,
                self.gp.bg_window,
                position_skip_mask=high_signal_mask
            )
            bg_sum_mappable = np.ma.masked_less(bg_sum_mappable, self.gp.min_mappable_bg)
            self.gp.logger.debug(f"Background mappable bases calculated for {self.chrom_name}")
            
            bg_sum = self.smooth_int_counts(agg_cutcounts, self.gp.bg_window, position_skip_mask=high_signal_mask)
            bg_sum_sq = self.smooth_int_counts(
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
        
    
    def smooth_int_counts(self, signal, window, position_skip_mask=None):
        return nan_moving_sum(
            signal,
            window=window,
            dtype=self.int_dtype,
            position_skip_mask=position_skip_mask
        )

    @ensure_contig_exists
    def normalize_density(self, density: ProcessorOutputData, total_cutcounts) -> ProcessorOutputData:
        if density is None:
            raise NoContigPresentError
        density.data_df['normalized_density'] = normalize_density(
            density.data_df['density'],
            total_cutcounts
        )
        density.data_df.drop(columns=['density'], inplace=True)
        return density