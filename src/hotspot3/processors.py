import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pandas as pd
from typing import Iterable, List, cast

from genome_tools.genomic_interval import GenomicInterval

from hotspot3.models import ProcessorOutputData, NotEnoughDataForContig

from hotspot3.io import run_bam2_bed
from hotspot3.io.logging import WithLoggerAndInterval, WithLogger
from hotspot3.io.readers import ChromReader, GenomeReader
from hotspot3.io.writers import ChromWriter, GenomeWriter

from hotspot3.connectors.babachi import BabachiWrapper

from hotspot3.signal_smoothing import modwt_smooth, normalize_density
from hotspot3.background_fit.segment_fit import SegmentFit, ChromosomeFit
from hotspot3.scoring.pvalue import PvalueEstimator
from hotspot3.peak_calling.peak_calling import find_stretches, find_varwidth_peaks

from hotspot3.utils import is_iterable, ensure_contig_exists


class GenomeProcessor(WithLogger):
    """
    Main class to run hotspot3-related functions and store parameters.

    Parameters:
        - chrom_sizes: Dictionary containing chromosome sizes.
        - mappable_bases_file: Path to the tabix-indexed file containing mappable bases or None.
        - tmp_dir: Temporary directory for intermediate files. Will use system default if None.
        - chromosomes: List of chromosomes to process or None. Used mostly for debugging. Will generate FDR corrections only for these chromosomes.

        - config: ProcessorConfig object containing parameters.
    """
    def __init__(self, chrom_sizes, config=None, mappable_bases_file=None, chromosomes=None):
        super().__init__(config=config)
        self.mappable_bases_file = mappable_bases_file
        self.chrom_sizes = chrom_sizes
        self.cpus = min(self.config.cpus, max(1, mp.cpu_count()))

        if chromosomes is not None:
            self.chrom_sizes = {k: v for k, v in chrom_sizes.items() if k in chromosomes}
        
        chroms = [x for x in self.chrom_sizes.keys()]
        self.logger.debug(f"Chromosomes to process: {chroms}")

        self.reader = self.copy_with_params(GenomeReader)
        self.writer = self.copy_with_params(GenomeWriter)
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

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

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
        return ProcessorOutputData('all', data)

    # Processing functions
    def write_cutcounts(self, bam_path, outpath):
        # TODO: rewrite with pysam
        self.logger.info('Extracting cutcounts from bam file')
        run_bam2_bed(bam_path, outpath, self.chrom_sizes.keys())

    def get_total_cutcounts(self, cutcounts_path) -> int:
        total_cutcounts = sum(
            self.parallel_by_chromosome(
                ChromosomeProcessor.get_total_cutcounts,
                cutcounts_path
            )
        )
        self.logger.info('Total cutcounts = %d', total_cutcounts)
        return total_cutcounts


    def smooth_signal_modwt(self, cutcounts_path, save_path, total_cutcounts_path):
        self.logger.info('Smoothing signal using MODWT')
        total_cutcounts = self.get_total_cutcounts(cutcounts_path)
        self.writer.save_cutcounts(total_cutcounts, total_cutcounts_path)

        self.writer.clean_path(save_path)
        self.parallel_by_chromosome(
            ChromosomeProcessor.smooth_density_modwt,
            cutcounts_path,
            total_cutcounts,
            save_path
        )
    

    def fit_background_model(self, cutcounts_file, save_path: str, per_region_stats_path):
        self.logger.info('Estimating parameters of background model')
        
        self.writer.clean_path(save_path)
        per_region_params = self.parallel_by_chromosome(
            ChromosomeProcessor.fit_background_model,
            cutcounts_file,
            save_path,
        )
        per_region_params = self.merge_and_add_chromosome(per_region_params).data_df
        cols_order = ['chrom'] + [col for col in per_region_params.columns if col != 'chrom']
        per_region_params = per_region_params[cols_order]
        self.writer.df_to_gzip(per_region_params, per_region_stats_path)


    def calc_pvals(self, cutcounts_file, fit_path: str, save_path: str):
        self.logger.info('Calculating per-bp p-values')
        
        self.writer.clean_path(save_path)
        self.parallel_by_chromosome(
            ChromosomeProcessor.calc_pvals,
            cutcounts_file,
            fit_path,
            save_path
        )

    
    def calc_fdr(self, pvals_path, save_path, max_fdr=1):
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
        self.writer.clean_path(save_path)
        self.parallel_by_chromosome(
            ChromosomeProcessor.write_to_parquet,
            result,
            save_path,
            0
        )


    def call_hotspots(self, fdr_path, sample_id, save_path, fdr_tr=0.05):
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
        )
        hotspots = self.merge_and_add_chromosome(hotspots)
        hotspots.data_df['id'] = sample_id
        hotspots.data_df['score'] = np.round(hotspots.data_df['max_neglog10_fdr'] * 10).astype(np.int64).clip(0, 1000)
        if len(hotspots.data_df) == 0:
            self.logger.critical(f"No hotspots called at FDR={fdr_tr}. Most likely something went wrong!")
        else:
            self.logger.info(f"There are {len(hotspots.data_df)} hotspots called at FDR={fdr_tr}")
        
        self.writer.df_to_tabix(hotspots, save_path)


    def call_variable_width_peaks(
            self,
            smoothed_signal_path,
            fdrs_path,
            sample_id,
            save_path,
            fdr_tr=0.05
        ):
        """
        Call variable width peaks from smoothed signal and hotspots.

        Parameters:
            - smoothed_signal_path: Path to the parquet file containing the smoothed signal.
            - fdrs_path: Path to the parquet file containing the log10(FDR) values.
            - fdr_tr: FDR threshold for calling peaks.
        
        Returns:
            - peaks_data: ProcessorOutputData containing the peaks in bed format

        """
        if sample_id is None:
            sample_id = 'id'
        peaks_data = self.parallel_by_chromosome(
            ChromosomeProcessor.call_variable_width_peaks,
            smoothed_signal_path,
            fdrs_path,
            fdr_tr
        )
        peaks_data = self.merge_and_add_chromosome(peaks_data).data_df
        if len(peaks_data) == 0:
            self.logger.critical(f"No peaks called at FDR={fdr_tr}. Most likely something went wrong!")
            return
        else:
            self.logger.info(f"There are {len(peaks_data)} peaks called at FDR={fdr_tr}")
        peaks_data['id'] = sample_id
        peaks_data = peaks_data[['chrom', 'start', 'end', 'id', 'max_density', 'summit']]
        self.writer.df_to_tabix(peaks_data, save_path)


    def extract_normalized_density(self, smoothed_signal, density_path):
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


    @ensure_contig_exists
    def get_total_cutcounts(self, cutcounts) -> int:
        """
        Get total # of cutcounts for the chromosome.
        """
        self.logger.debug(f"{self.chrom_name}: Calculating total cutcounts")
        return self.reader.extract_cutcounts(cutcounts).sum()


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
        
        s_fit = self.copy_with_params(SegmentFit)
        per_window_trs_global, rmseas, global_fit_params = s_fit.fit_segment_thresholds(
            agg_cutcounts,
        )
        
        if global_fit_params.rmsea > self.config.rmsea_tr:
            self.logger.warning(f"{self.chrom_name}: Best RMSEA: {global_fit_params.rmsea:.3f}. Chromosome fit might be poorly approximated.")

        self.logger.debug(f"{self.chrom_name}: Signal quantile: {global_fit_params.fit_quantile:.3f}. signal threshold: {global_fit_params.fit_threshold:.0f}. Best RMSEA: {global_fit_params.rmsea:.3f}")

        good_fits_n = np.sum(rmseas <= self.config.rmsea_tr)
        n_rmsea = np.sum(~np.isnan(rmseas))
        self.logger.debug(f"{self.chrom_name}: Signal thresholds approximated. {good_fits_n:,}/{n_rmsea:,} strided windows have RMSEA <= {self.config.rmsea_tr:.2f}")

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

        chrom_fit = self.copy_with_params(ChromosomeFit)
        fit_res, per_window_trs, final_rmsea, per_interval_params = chrom_fit.fit_params(
            agg_cutcounts=agg_cutcounts,
            bad_segments=bad_segments,
            global_fit=global_fit_params
        )

        df = pd.DataFrame({
            'sliding_r': fit_res.r,
            'sliding_p': fit_res.p,
            'tr': per_window_trs,
            'initial_tr': per_window_trs_global,
            'bad': segmentation.annotate_with_segments(agg_cutcounts.shape, bad_segments),
            'rmsea': final_rmsea,
            'enough_bg': fit_res.enough_bg_mask
        })
        self.write_to_parquet(df, save_path, compression_level=0)
        
        return ProcessorOutputData(self.chrom_name, per_interval_params)

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


    @ensure_contig_exists
    def call_hotspots(self, fdr_path, fdr_threshold=0.05) -> ProcessorOutputData:
        log10_fdrs = self.reader.extract_fdr_track(fdr_path)
        self.logger.debug(f"{self.chrom_name}: Calling hotspots")
        signif_fdrs = (log10_fdrs >= -np.log10(fdr_threshold))
        smoothed_signif = self.reader.bn_wrapper.centered_running_nansum(
            signif_fdrs,
            window=self.config.window,
        ) > 0
        region_starts, region_ends = find_stretches(smoothed_signif)

        max_log10_fdrs = np.empty(region_ends.shape)
        for i in range(len(region_starts)):
            start = region_starts[i]
            end = region_ends[i]
        
            max_log10_fdrs[i] = np.nanmax(log10_fdrs[start:end])
        
        self.logger.debug(f"{self.chrom_name}: {len(region_starts)} hotspots called")

        data = pd.DataFrame({
            'start': region_starts,
            'end': region_ends,
            'max_neglog10_fdr': max_log10_fdrs
        })
        return ProcessorOutputData(self.chrom_name, data)
    

    @ensure_contig_exists
    def call_variable_width_peaks(self, smoothed_signal_path, fdr_path, fdr_threshold) -> ProcessorOutputData:
        signal_df = self.reader.extract_from_parquet(
            smoothed_signal_path,
            columns=['smoothed', 'normalized_density']
        )
        self.logger.debug(f"{self.chrom_name}: Calling peaks at FDR={fdr_threshold}")
        signif_fdrs = self.reader.extract_fdr_track(fdr_path) >= -np.log10(fdr_threshold)
        starts, ends = find_stretches(signif_fdrs)
        if len(starts) == 0:
            self.logger.warning(f"{self.chrom_name}: No peaks found. Skipping...")
            raise NotEnoughDataForContig

        normalized_density = signal_df['normalized_density'].values
        peaks_in_hotspots_trimmed, _ = find_varwidth_peaks(
            signal_df['smoothed'].values,
            starts,
            ends
        )
        peaks_df = pd.DataFrame(
            peaks_in_hotspots_trimmed,
            columns=['start', 'summit', 'end']
        )
        peaks_df['summit_density'] = normalized_density[peaks_df['summit']]
        
        peaks_df['max_density'] = [
            np.max(normalized_density[start:end])
            for start, end in zip(peaks_df['start'], peaks_df['end'])
        ]
        self.logger.debug(f"{self.chrom_name}: {len(peaks_df)} peaks called")

        return ProcessorOutputData(self.chrom_name, peaks_df)


    @ensure_contig_exists
    def extract_normalized_density(self, smoothed_signal) -> ProcessorOutputData:
        data_df = self.reader.extract_from_parquet(
            smoothed_signal, 
            columns=['chrom', 'normalized_density']
        )[::self.config.density_step]
        
        data_df['start'] = np.arange(len(data_df)) * self.config.density_step
        data_df.query('normalized_density > 0', inplace=True)
        return ProcessorOutputData(self.chrom_name, data_df)


    @ensure_contig_exists
    def write_to_parquet(self, data_df, path, compression_level=22):
        self.writer.parallel_write_to_parquet(
            data_df,
            path,
            chrom_names=[x for x in self.gp.chrom_sizes.keys()],
            compression_level=compression_level,
        )
