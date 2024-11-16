import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pandas as pd
from typing import Iterable, List, cast
import functools

from genome_tools.genomic_interval import GenomicInterval

from hotspot3.models import ProcessorOutputData, NotEnoughDataForContig

from hotspot3.io.logging import WithLoggerAndInterval, WithLogger
from hotspot3.io.readers import ChromReader, GenomeReader
from hotspot3.io.writers import ChromWriter, GenomeWriter

from hotspot3.connectors.babachi import BabachiWrapper

from hotspot3.signal_smoothing import modwt_smooth, normalize_density
from hotspot3.background_fit.segment_fit import ChromosomeFit, SegmentsFit
from hotspot3.scoring.pvalue import PvalueEstimator
from hotspot3.peak_calling import find_stretches, find_varwidth_peaks
from hotspot3.utils import is_iterable, ensure_contig_exists
from hotspot3.stats import upper_bg_quantile


def parallel_func_error_handler(func):
    @functools.wraps(func)
    def wrapper(processor: 'ChromosomeProcessor', *args, **kwargs):
        try:
            return func(processor, *args, **kwargs)
        except:
            processor.logger.exception(f"Exception occured in {func.__name__} for chromosome {processor.chrom_name}")
            raise
    return wrapper


class GenomeProcessor(WithLogger):
    """
    Main class to run hotspot3-related functions and store parameters.

    Parameters:
        - chrom_sizes: Path to chromosome sizes file.
        - mappable_bases_file: Path to the tabix-indexed file containing mappable bases or None.
        - tmp_dir: Temporary directory for intermediate files. Will use system default if None.
        - chromosomes: List of chromosomes to process or None. Used mostly for debugging. Will generate FDR corrections only for these chromosomes.

        - config: ProcessorConfig object containing parameters.
    """
    def __init__(self, chrom_sizes_file=None, config=None, mappable_bases_file=None, chromosomes=None):
        super().__init__(config=config)
        self.reader = self.copy_with_params(GenomeReader)
        self.writer = self.copy_with_params(GenomeWriter)
        self.mappable_bases_file = mappable_bases_file
        self.chrom_sizes_file = chrom_sizes_file

        self.cpus = min(self.config.cpus, max(1, mp.cpu_count()))

        self.chrom_sizes = self.reader.read_chrom_sizes(chrom_sizes_file)

        if chromosomes is not None:
            self.chrom_sizes = {k: v for k, v in self.chrom_sizes.items() if k in chromosomes}
        
        chroms = [x for x in self.chrom_sizes.keys()]
        self.logger.debug(f"Chromosomes to process: {chroms}")


        chrom_intervals = [
            GenomicInterval(chrom, 0, length) 
            for chrom, length in self.chrom_sizes.items()
        ]
        self.chromosome_processors = sorted(
            [
                ChromosomeProcessor(self, interval, config=self.config, logger=self.logger) 
                for interval in chrom_intervals
            ],
            key=lambda x: x.chrom_size,
            reverse=True
        )
    
    def __getstate__(self):
        state = self.__dict__
        if 'logger' in state:
            del state['logger']
        return state

    def __setstate__(self, state: dict):
        for name, value in state.items():
            setattr(self, name, value)
    
    ## Parallel processing functions

    def construct_parallel_args(self, *args):
        res_args = []
        for arg in args:
            if is_iterable(arg):
                if all(isinstance(x, ProcessorOutputData) for x in arg):
                     # if arg consits of ProcessorOutputData - 
                     # sort by chromosome name to match chromosome_processors
                    arg = cast(List[ProcessorOutputData], arg)
                    tmp = {x.id: x for x in arg}
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
        self.logger.debug(f'Using {cpus} CPUs to {func.__name__}')
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
                    self.logger.critical(f"Exception occured, gracefully shutting down executor...")
                    self.logger.critical(e)
                    raise e

        self.logger.debug(f'Results of {func.__name__} emitted.')
        return results
    
    def merge_and_add_chromosome(self, results: Iterable[ProcessorOutputData]) -> ProcessorOutputData:
        data = []
        results = sorted(results, key=lambda x: x.id)
        categories = [x.id for x in results]
        for res in results:
            df = res.data_df
            df['chrom'] = pd.Categorical(
                [res.id] * df.shape[0],
                categories=categories,
            )
            data.append(df)
            
        data = pd.concat(data, ignore_index=True)
        data = data[['chrom'] + [col for col in data.columns if col != 'chrom']]
        return ProcessorOutputData('all', data)


    ## Processing functions
    def extract_cutcounts_from_bam(self, bam_path, outpath):
        """
        Extract cutcounts from BAM file and save to tabix file.

        Parameters:
            - bam_path: Path to the BAM/CRAM/SAM file.
            - outpath: Path to save the cutcounts to.
        """
        self.logger.info('Extracting cutcounts from bam file')
        if self.cpus >= 3:
            data = self.parallel_by_chromosome(
                ChromosomeProcessor.extract_cutcounts_for_chromosome,
                bam_path,
            )
            data = self.merge_and_add_chromosome(data).data_df
            self.writer.df_to_tabix(data, outpath)
        else:
            # BAM2BED is faster in ~almost~ single-threaded mode
            self.reader.extract_cutcounts_all_chroms(
                bam_path,
                outpath,
                self.chrom_sizes.keys()
            )


    def get_total_cutcounts(self, cutcounts_path, total_cutcounts_path) -> int:
        """
        Extract total cutcounts from cutcounts file and save to a separate file.

        Parameters:
            - cutcounts_path: Path to tabix file containing the cutcounts.
            - total_cutcounts_path: Path to save the total cutcounts to.
        """
        total_cutcounts = sum(
            self.parallel_by_chromosome(
                ChromosomeProcessor.get_total_cutcounts,
                cutcounts_path
            )
        )
        self.logger.info('Total cutcounts = %d', total_cutcounts)
        self.writer.save_cutcounts(total_cutcounts, total_cutcounts_path)


    def smooth_signal_modwt(self, cutcounts_path, save_path, total_cutcounts_path):
        """
        Smooth signal using MODWT and save to parquet file.
        """
        self.logger.info('Smoothing signal using MODWT')
        total_cutcounts = self.reader.read_total_cutcounts(total_cutcounts_path)
        
        self.writer.sanitize_path(save_path)
        self.parallel_by_chromosome(
            ChromosomeProcessor.smooth_density_modwt,
            cutcounts_path,
            total_cutcounts,
            save_path
        )

    def extract_thresholds_to_bw(self, fit_params_path, total_cutcounts_path, save_path):
        """
        Convert threshold values to bigwig file.
        """
        self.logger.info('Converting threshold values to bigwig')
        thresholds = self.parallel_by_chromosome(
            ChromosomeProcessor.extract_fit_threholds,
            fit_params_path
        )
        total_cutcounts = self.reader.read_total_cutcounts(total_cutcounts_path)
        thresholds = self.merge_and_add_chromosome(thresholds).data_df
        thresholds['tr'] = normalize_density(thresholds['tr'], total_cutcounts)
        self.writer.df_to_bigwig(
            thresholds,
            save_path,
            self.chrom_sizes,
            col='tr'
        )
    
    def extract_bg_quantile_to_bw(self, fit_params_path, total_cutcounts_path, save_path):
        """
        Convert threshold values to bigwig file.
        """
        self.logger.info('Converting background quantile values to bigwig')
        thresholds = self.parallel_by_chromosome(
            ChromosomeProcessor.extract_bg_quantile,
            fit_params_path
        )
        total_cutcounts = self.reader.read_total_cutcounts(total_cutcounts_path)
        thresholds = self.merge_and_add_chromosome(thresholds).data_df
        thresholds['tr'] = upper_bg_quantile(thresholds['r'], thresholds['p'])
        thresholds['tr'] = normalize_density(thresholds['tr'], total_cutcounts)
        self.writer.df_to_bigwig(
            thresholds,
            save_path,
            self.chrom_sizes,
            col='tr'
        )
    

    def fit_background_model(self, cutcounts_file, save_path: str, per_region_stats_path):
        """"
        Fit background model of cutcounts distribution and save fit parameters to parquet file.
        """
        self.logger.info('Estimating parameters of background model')
        
        self.writer.sanitize_path(save_path)
        per_region_params = self.parallel_by_chromosome(
            ChromosomeProcessor.fit_background_model,
            cutcounts_file,
            save_path,
        )
        per_region_params = self.merge_and_add_chromosome(per_region_params).data_df
        self.writer.df_to_tabix(per_region_params, per_region_stats_path)


    def calc_pvals(self, cutcounts_file, fit_path: str, save_path: str):
        """
        Calculate per-bp p-values from cutcounts and background model parameters.
        """
        self.logger.info('Calculating per-bp p-values')
        
        self.writer.sanitize_path(save_path)
        self.parallel_by_chromosome(
            ChromosomeProcessor.calc_pvals,
            cutcounts_file,
            fit_path,
            save_path
        )

    
    def calc_fdr(self, pvals_path, save_path, max_fdr=1):
        """
        Calculate fast per-bp FDRs from p-values. If max_fdr is set to 1, will calculate all FDRs.

        Parameters:
            - pvals_path: Path to the parquet file containing the p-values.
            - save_path: Path to save the FDRs in parquet format.
            - max_fdr: Maximum FDR to calculate.
        """
        self.logger.info('Calculating per-bp FDRs')
        
        chrom_pos_mapping = self.reader.read_chrom_pos_mapping(pvals_path)

        pval_est = PvalueEstimator(config=self.config, logger=self.logger)
        result = pval_est.log10_fdr_from_log10pvals(
            self.reader.read_pval_from_parquet(pvals_path),
            max_fdr,
        )

        result = [
            ProcessorOutputData(
                chrom, 
                pd.DataFrame({'log10_fdr': result[start:end]})
            )
            for chrom, start, end
            in zip(*chrom_pos_mapping)
        ]
        self.logger.debug('Saving per-bp FDRs')
        self.writer.sanitize_path(save_path)
        self.parallel_by_chromosome(
            ChromosomeProcessor.write_to_parquet,
            result,
            save_path,
            0
        )


    def call_hotspots(self, fdr_path, sample_id, save_path, save_path_bb, fdr_tr):
        """
        Call hotspots from parquet file containing log10(FDR) values.

        Parameters:
            - fdr_path: Path to the parquet file containing the log10(FDR) values.
            - sample_id: Sample ID.
            - save_path: Path to save the hotspots in bed format.
            - save_path_bb: Path to save the hotspots in bigbed format.
            - fdr_tr: FDR threshold for calling hotspots.
        """
        hotspots = self.parallel_by_chromosome(
            ChromosomeProcessor.call_hotspots,
            fdr_path,
            fdr_tr,
        )
        hotspots = self.merge_and_add_chromosome(hotspots).data_df
        signif_stretches = hotspots['signif_stretches'].values
        hotspots['id'] = sample_id
        if len(hotspots) == 0:
            self.logger.critical(f"No hotspots called at FDR={fdr_tr}. Most likely something went wrong!")
        else:
            self.logger.info(f"There are {len(hotspots)} hotspots called at FDR={fdr_tr}")
        
        hotspots = hotspots[['chrom', 'start', 'end', 'id', 'max_neglog10_fdr']]
        self.writer.df_to_tabix(hotspots, save_path)

        hotspots = self.writer.hotspots_to_bed12(hotspots, fdr_tr, signif_stretches)
        self.writer.df_to_bigbed(hotspots, self.chrom_sizes_file, save_path_bb)


    def call_variable_width_peaks(
            self,
            smoothed_signal_path,
            fdrs_path,
            total_cutcounts_path,
            sample_id,
            save_path,
            save_path_bb,
            fdr_tr=0.05
        ):
        """
        Call variable width peaks from smoothed signal and parquet file containing log10(FDR) values.

        Parameters:
            - smoothed_signal_path: Path to the parquet file containing the smoothed signal.
            - fdrs_path: Path to the parquet file containing the log10(FDR) values.
            - sample_id: Sample ID.
            - save_path: Path to save the peaks in bed format.
            - save_path_bb: Path to save the peaks in bigbed format.
            - fdr_tr: FDR threshold for calling peaks.
        """
        peaks_data = self.parallel_by_chromosome(
            ChromosomeProcessor.call_variable_width_peaks,
            smoothed_signal_path,
            fdrs_path,
            fdr_tr
        )
        total_cutcounts = self.reader.read_total_cutcounts(total_cutcounts_path)
        peaks = self.merge_and_add_chromosome(peaks_data).data_df
        if len(peaks) == 0:
            self.logger.critical(f"No peaks called at FDR={fdr_tr}. Most likely something went wrong!")
            return
        else:
            self.logger.info(f"There are {len(peaks)} peaks called at FDR={fdr_tr}")
        
        peaks['id'] = sample_id
        peaks = peaks[['chrom', 'start', 'end', 'id', 'max_density', 'summit_density', 'summit', 'smoothed_peak_height']]
        peaks['smoothed_peak_height'] = normalize_density(
            peaks['smoothed_peak_height'],
            total_cutcounts
        )
        self.writer.df_to_tabix(peaks, save_path)

        peaks = self.writer.peaks_to_bed12(peaks, fdr_tr)
        self.writer.df_to_bigbed(peaks, self.chrom_sizes_file, save_path_bb)


    def extract_normalized_density(self, smoothed_signal, density_path):
        """
        Save normalized density to bigwig file. Uses the density_step parameter from the config to downsample the data.

        Parameters:
            - smoothed_signal: Path to the parquet file containing the smoothed signal.
            - density_path: Path to save the bigwig file.
        """
        density_data = self.parallel_by_chromosome(
            ChromosomeProcessor.extract_normalized_density,
            smoothed_signal
        )
        density_data = self.merge_and_add_chromosome(density_data).data_df
        density_data['end'] = density_data['start'] + self.config.density_step
        self.logger.debug(f"Converting density to bigwig")
        self.writer.df_to_bigwig(
            density_data,
            density_path,
            self.chrom_sizes,
            col='normalized_density'
        )
        

class ChromosomeProcessor(WithLoggerAndInterval):
    """
    Individual chromosome processor. Used for parallel processing of chromosomes.
    Don't use directly, use GenomeProcessor instead.
    """
    def __init__(self, genome_processor: GenomeProcessor, chromosome_interval: GenomicInterval, config=None, logger=None) :
        super().__init__(chromosome_interval, config=config, logger=logger)
        self.chrom_name = self.genomic_interval.chrom
        self.gp = genome_processor
        
        self.chrom_size = len(self.genomic_interval)
        self.reader = self.copy_with_params(ChromReader)
        self.writer = self.copy_with_params(ChromWriter)


    @parallel_func_error_handler
    @ensure_contig_exists
    def extract_cutcounts_for_chromosome(self, bam_path) -> ProcessorOutputData:
        data = self.reader.extract_cutcounts_for_chrom(bam_path)
        return ProcessorOutputData(self.chrom_name, data)


    @parallel_func_error_handler
    @ensure_contig_exists
    def get_total_cutcounts(self, cutcounts) -> int:
        """
        Get total # of cutcounts for the chromosome.
        """
        self.logger.debug(f"{self.chrom_name}: Calculating total cutcounts")
        return self.reader.extract_cutcounts(cutcounts).sum()


    @parallel_func_error_handler
    @ensure_contig_exists
    def smooth_density_modwt(self, cutcounts_path, total_cutcounts, save_path):
        """
        Run MODWT smoothing on cutcounts. Smoothed density will be used to call peaks.
        """
        self.logger.debug(f"{self.chrom_name}: Running modwt signal smoothing (filter={self.config.filter}, level={self.config.modwt_level})")
        agg_cutcounts = self.reader.extract_aggregated_cutcounts(cutcounts_path)
        smoothed = modwt_smooth(agg_cutcounts, self.config.filter, level=self.config.modwt_level)
        data = pd.DataFrame({
            'smoothed': smoothed,
            'normalized_density': normalize_density(agg_cutcounts, total_cutcounts) 
        })
        self.write_to_parquet(data, save_path)
    

    @parallel_func_error_handler
    @ensure_contig_exists
    def fit_background_model(self, cutcounts_file, save_path) -> ProcessorOutputData:
        """
        Fit background model to cutcounts and save fit parameters to parquet file.
        """
        agg_cutcounts = self.reader.extract_mappable_agg_cutcounts(
            cutcounts_file,
            self.gp.mappable_bases_file
        )
        self.logger.debug(f'{self.chrom_name}: Estimating proportion of background vs signal')
        
        s_fit = self.copy_with_params(ChromosomeFit) # Change to GlobalFit
        per_window_trs_global, global_fit_params = s_fit.fit_segment_thresholds(
            agg_cutcounts,
        )
        
        if global_fit_params.rmsea > self.config.rmsea_tr:
            self.logger.warning(f"{self.chrom_name}: Best RMSEA: {global_fit_params.rmsea:.3f}. Chromosome fit might be poorly approximated.")

        self.logger.debug(f"{self.chrom_name}: Signal quantile: {global_fit_params.fit_quantile:.3f}. signal threshold: {global_fit_params.fit_threshold:.0f}. Best RMSEA: {global_fit_params.rmsea:.3f}")

        # Segmentation
        segmentation = self.copy_with_params(BabachiWrapper)
        bad_segments = segmentation.run_segmentation(
            agg_cutcounts,
            per_window_trs_global,
            global_fit_params
        )

        self.logger.debug(
            f'{self.chrom_name}: Estimating per-bp parameters of background model for {len(bad_segments)} segments'
        )

        chrom_fit = self.copy_with_params(SegmentsFit)
        fit_res, per_window_trs, final_rmsea, per_interval_params = chrom_fit.fit_params(
            agg_cutcounts=agg_cutcounts,
            bad_segments=bad_segments,
            fallback_fit_results=global_fit_params
        )

        df = pd.DataFrame({
            'sliding_r': fit_res.r,
            'sliding_p': fit_res.p,
            'enough_bg': fit_res.enough_bg_mask,
            'tr': per_window_trs,
            'initial_tr': per_window_trs_global,
            'bad': segmentation.annotate_with_segments(agg_cutcounts.shape, bad_segments),
            'rmsea': final_rmsea,
        })
        self.write_to_parquet(df, save_path, compression_level=0)
        
        return ProcessorOutputData(self.chrom_name, per_interval_params)


    @parallel_func_error_handler
    @ensure_contig_exists
    def extract_fit_threholds(self, fit_parquet_path) -> ProcessorOutputData:
        fit_res = self.reader.extract_fit_threholds(fit_parquet_path).iloc[::self.config.density_step]
        fit_res['start'] = np.arange(len(fit_res)) * self.config.density_step
        fit_res['end'] = fit_res['start'] + self.config.density_step
        return ProcessorOutputData(self.chrom_name, fit_res)

    @parallel_func_error_handler
    @ensure_contig_exists
    def extract_bg_quantile(self, fit_parquet_path) -> ProcessorOutputData:
        fit_res = self.reader.extract_fit_params(fit_parquet_path)
        fit_res = pd.DataFrame({
            'r': fit_res.r,
            'p': fit_res.p,
        }).iloc[::self.config.density_step].fillna(0)
        fit_res['start'] = np.arange(len(fit_res)) * self.config.density_step
        fit_res['end'] = fit_res['start'] + self.config.density_step
        return ProcessorOutputData(self.chrom_name, fit_res)

    @parallel_func_error_handler
    @ensure_contig_exists
    def calc_pvals(self, cutcounts_file, fit_parquet_path, save_path) -> ProcessorOutputData:
        self.logger.debug(f'{self.chrom_name}: Calculating p-values')
        agg_cutcounts = self.reader.extract_mappable_agg_cutcounts(
            cutcounts_file,
            self.gp.mappable_bases_file
        )
        agg_cutcounts = np.floor(agg_cutcounts.filled(np.nan))
        fit_res = self.reader.extract_fit_params(fit_parquet_path)
    
        pval_estimator = self.copy_with_params(PvalueEstimator)

        neglog_pvals = pval_estimator.estimate_pvalues(agg_cutcounts, fit_res)
        self.logger.debug(f"Saving p-values for {self.chrom_name}")
 
        neglog_pvals = pd.DataFrame({'log10_pval': pval_estimator.fix_inf_pvals(neglog_pvals)})
        self.write_to_parquet(neglog_pvals, save_path)


    @parallel_func_error_handler
    @ensure_contig_exists
    def call_hotspots(self, fdr_path, fdr_threshold=0.05) -> ProcessorOutputData:
        log10_fdrs = self.reader.extract_fdr_track(fdr_path)
        self.logger.debug(f"{self.chrom_name}: Calling hotspots")
        
        # TODO: move logic to a peak calling class
        signif_fdrs = (log10_fdrs >= -np.log10(fdr_threshold))
        smoothed_signif = self.reader.extract_significant_bases(
            log10_fdrs,
            fdr_threshold,
            min_signif_bases=1
        )
        region_starts, region_ends = find_stretches(smoothed_signif)

        max_log10_fdrs = np.empty(region_ends.shape)
        signif_stretches = []
        n_signif_bases = []
        for i in range(len(region_starts)):
            start = region_starts[i]
            end = region_ends[i]
            max_log10_fdrs[i] = np.nanmax(log10_fdrs[start:end])
            signif_stretches.append(find_stretches(signif_fdrs[start:end]))
            n_signif_bases.append(np.sum(signif_fdrs[start:end]))

        self.logger.debug(f"{self.chrom_name}: {len(region_starts)} hotspots called")

        data = pd.DataFrame({
            'start': region_starts,
            'end': region_ends,
            'max_neglog10_fdr': max_log10_fdrs,
            'n_significant_bases': n_signif_bases,
            'signif_stretches': signif_stretches,
        })
        return ProcessorOutputData(self.chrom_name, data)


    @parallel_func_error_handler
    @ensure_contig_exists
    def call_variable_width_peaks(self, smoothed_signal_path, fdr_path, fdr_threshold) -> ProcessorOutputData:
        signal_df = self.reader.extract_from_parquet(
            smoothed_signal_path,
            columns=['smoothed', 'normalized_density']
        )
        log10_fdrs = self.reader.extract_fdr_track(fdr_path)
        self.logger.debug(f"{self.chrom_name}: Calling peaks at FDR={fdr_threshold}")
        
        # TODO: move logic to a peak calling class
        smoothed_signif = self.reader.extract_significant_bases(
            log10_fdrs,
            fdr_threshold,
            min_signif_bases=self.config.min_signif_bases_for_peak
        )

        starts, ends = find_stretches(smoothed_signif)
        if len(starts) == 0:
            self.logger.warning(f"{self.chrom_name}: No peaks found. Skipping...")
            raise NotEnoughDataForContig

        normalized_density = signal_df['normalized_density'].values
        peaks_in_hotspots_trimmed, cutoff_height = find_varwidth_peaks(
            signal_df['smoothed'].values,
            starts,
            ends
        )
        peaks_df = pd.DataFrame(
            peaks_in_hotspots_trimmed,
            columns=['start', 'summit', 'end']
        )
        # TODO: wrap in function
        peaks_df['end'] = peaks_df['summit'] + np.maximum(peaks_df.eval('end - summit'), self.config.min_peak_half_width)
        peaks_df['start'] = peaks_df['summit'] - np.maximum(peaks_df.eval('summit - start'), self.config.min_peak_half_width)
    
        peaks_df['end'] += 1 # BED format, half-open intervals
        peaks_df['summit_density'] = normalized_density[peaks_df['summit']]
        peaks_df['smoothed_peak_height'] = cutoff_height
        
        peaks_df['max_density'] = [
            np.max(normalized_density[start:end])
            for start, end in zip(peaks_df['start'], peaks_df['end'])
        ]
        self.logger.debug(f"{self.chrom_name}: {len(peaks_df)} peaks called")

        return ProcessorOutputData(self.chrom_name, peaks_df)


    @parallel_func_error_handler
    @ensure_contig_exists
    def extract_normalized_density(self, smoothed_signal) -> ProcessorOutputData:
        data_df = self.reader.extract_from_parquet(
            smoothed_signal, 
            columns=['chrom', 'normalized_density']
        )[::self.config.density_step]
        
        data_df['start'] = np.arange(len(data_df)) * self.config.density_step
        data_df.query('normalized_density > 0', inplace=True)
        return ProcessorOutputData(self.chrom_name, data_df)


    @parallel_func_error_handler
    @ensure_contig_exists
    def write_to_parquet(self, data_df, path, compression_level=22):
        self.writer.parallel_write_to_parquet(
            data_df,
            path,
            chrom_names=[x for x in self.gp.chrom_sizes.keys()],
            compression_level=compression_level,
        )
