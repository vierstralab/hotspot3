import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pandas as pd
import gc
from typing import Iterable
import tempfile
import shutil
import os
import subprocess
import importlib.resources as pkg_resources
from genome_tools.genomic_interval import GenomicInterval

from hotspot3.logging import setup_logger
from hotspot3.models import ProcessorOutputData, NotEnoughDataForContig, ProcessorConfig
from hotspot3.file_extractors import ChromosomeExtractor
from hotspot3.pvalue import PvalueEstimator
from hotspot3.connectors.bottleneck import BottleneckWrapper
from hotspot3.connectors.babachi import BabachiWrapper
from hotspot3.segment_fit import SegmentFit, ChromosomeFit
from hotspot3.signal_smoothing import modwt_smooth
from hotspot3.peak_calling import find_stretches, find_varwidth_peaks
from hotspot3.stats import fast_logfdr_below_threshold, fix_inf_pvals
from hotspot3.utils import normalize_density, is_iterable, to_parquet_high_compression, delete_path, df_to_bigwig, ensure_contig_exists


def run_bam2_bed(bam_path, tabix_bed_path, chromosomes=None):
    """
    Run bam2bed conversion script.
    """
    with pkg_resources.path('hotspot3.scripts', 'extract_cutcounts.sh') as script:
        chroms = ','.join(chromosomes) if chromosomes else ""
        subprocess.run(
            ['bash', script, bam_path, tabix_bed_path, chroms],
            check=True,
            text=True
        )


class GenomeProcessor:
    """
    Main class to run hotspot3-related functions and store parameters.

    Parameters:
        - chrom_sizes: Dictionary containing chromosome sizes.
        - mappable_bases_file: Path to the tabix-indexed file containing mappable bases or None.
        - tmp_dir: Temporary directory for intermediate files. Will use system default if None.
        - chromosomes: List of chromosomes to process or None. Used mostly for debugging. Will generate FDR corrections only for these chromosomes.

        - config: ProcessorConfig object containing parameters.
    """
    def __init__(self, chrom_sizes, config=None, mappable_bases_file=None, tmp_dir=None, chromosomes=None):
        if config is None:
            config = ProcessorConfig()
        self.config = config
        self.chrom_sizes = chrom_sizes
        if chromosomes is not None:
            self.chrom_sizes = {k: v for k, v in chrom_sizes.items() if k in chromosomes}
        
        chroms = [x for x in self.chrom_sizes.keys()]
        self.logger.debug(f"Chromosomes to process: {chroms}")
        self.mappable_bases_file = mappable_bases_file
        self.tmp_dir = tmp_dir
        self.cpus = min(self.config.cpus, max(1, mp.cpu_count()))

        self.chromosome_processors = sorted(
            [ChromosomeProcessor(self, chrom_name) for chrom_name in chroms],
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

    @property
    def logger(self) -> logging.Logger:
        return setup_logger(level=self.config.logger_level)


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
                    self.logger.critical(f"Exception occured, gracefully shutting down executor...")
                    self.logger.critical(e)
                    #exit(143)
                    raise e

        self.logger.debug(f'Results of {func.__name__} emitted.')
        return results

    # Processing functions
    def write_cutcounts(self, bam_path, outpath) -> None:
        # TODO: rewrite with pysam
        self.logger.info('Extracting cutcounts from bam file')
        run_bam2_bed(bam_path, outpath, self.chrom_sizes.keys())
    
    def total_cutcounts(self, cutcounts_path) -> int:
        total_cutcounts = sum(
            self.parallel_by_chromosome(
                ChromosomeProcessor.total_cutcounts,
                cutcounts_path
            )
        )
        self.logger.info('Total cutcounts = %d', total_cutcounts)
        return total_cutcounts

    def modwt_smooth_signal(self, cutcounts_path, total_cutcounts, save_path):
        self.logger.info('Smoothing signal using MODWT')
    
        delete_path(save_path)
        self.parallel_by_chromosome(
            ChromosomeProcessor.modwt_smooth_density,
            cutcounts_path,
            total_cutcounts,
            save_path
        )

    def calc_pval(self, cutcounts_file, pvals_path: str, fit_params_path: str) -> ProcessorOutputData:
        self.logger.info('Calculating per-bp p-values')
        delete_path(pvals_path)
        delete_path(fit_params_path)
        per_region_params = self.parallel_by_chromosome(
            ChromosomeProcessor.calc_pvals,
            cutcounts_file,
            pvals_path,
            fit_params_path
        )
        per_region_params = self.merge_and_add_chromosome(per_region_params)
        return per_region_params

    
    def calc_fdr(self, pvals_path, fdrs_path, max_fdr=1):
        self.logger.info('Calculating per-bp FDRs')
        chrom_pos_mapping = pd.read_parquet(
            pvals_path,
            engine='pyarrow', 
            columns=['chrom']
        )['chrom']

        total_len = chrom_pos_mapping.shape[0]
        chrom_pos_mapping = chrom_pos_mapping.drop_duplicates()
        starts = chrom_pos_mapping.index
        # file is always sorted within chromosomes
        ends = [*starts[1:], total_len]

        result = fast_logfdr_below_threshold(pvals_path, max_fdr, self.config.fdr_method)

        result = [
            ProcessorOutputData(
                chrom, 
                pd.DataFrame({'log10_fdr': result[start:end]})
            )
            for chrom, start, end
            in zip(chrom_pos_mapping, starts, ends)
        ]
        delete_path(fdrs_path)
        self.logger.debug('Saving per-bp FDRs')
        self.parallel_by_chromosome(
            ChromosomeProcessor.to_parquet,
            result,
            fdrs_path,
            0
        )
        return fdrs_path

    def call_hotspots(self, fdr_path, prefix, fdr_tr=0.05) -> ProcessorOutputData:
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
        hotspots.data_df['id'] = prefix
        hotspots.data_df['score'] = np.round(hotspots.data_df['max_neglog10_fdr'] * 10).astype(np.int64).clip(0, 1000)
        if len(hotspots.data_df) == 0:
            self.logger.critical(f"No hotspots called at FDR={fdr_tr}. Most likely something went wrong!")
        else:
            self.logger.info(f"There are {len(hotspots.data_df)} hotspots called at FDR={fdr_tr}")
        return hotspots

    def call_variable_width_peaks(self, smoothed_signal_path, fdrs_path, fdr_tr) -> ProcessorOutputData:
        """
        Call variable width peaks from smoothed signal and hotspots.

        Parameters:
            - smoothed_signal_path: Path to the parquet file containing the smoothed signal.
            - fdrs_path: Path to the parquet file containing the log10(FDR) values.
            - fdr_tr: FDR threshold for calling peaks.
        
        Returns:
            - peaks_data: ProcessorOutputData containing the peaks in bed format

        """
        peaks_data = self.parallel_by_chromosome(
            ChromosomeProcessor.call_variable_width_peaks,
            smoothed_signal_path,
            fdrs_path,
            fdr_tr
        )
        peaks_data = self.merge_and_add_chromosome(peaks_data)
        if len(peaks_data.data_df) == 0:
            self.logger.critical(f"No peaks called at FDR={fdr_tr}. Most likely something went wrong!")
        else:
            self.logger.info(f"There are {len(peaks_data.data_df)} peaks called at FDR={fdr_tr}")
        return peaks_data

    def extract_density(self, smoothed_signal, density_path):
        density_data = self.parallel_by_chromosome(
            ChromosomeProcessor.extract_density,
            smoothed_signal
        )
        density_data = self.merge_and_add_chromosome(density_data).data_df
        density_data['end'] = density_data['start'] + self.config.density_step
        self.logger.debug(f"Converting density to bigwig")
        df_to_bigwig(density_data, density_path, self.chrom_sizes, col='normalized_density')
        

class ChromosomeProcessor:
    """
    Individual chromosome processor. Used for parallel processing of chromosomes.
    Don't use directly, use GenomeProcessor instead.
    """
    def __init__(self, genome_processor: GenomeProcessor, chrom_name: str) -> None:
        self.chrom_name = chrom_name
        self.gp = genome_processor
        self.config = self.gp.config
        self.chrom_size = int(self.gp.chrom_sizes[chrom_name])
        self.genomic_interval = GenomicInterval(chrom_name, 0, self.chrom_size)
        self.extractor = ChromosomeExtractor(self.genomic_interval, config=self.config)

    @ensure_contig_exists
    def calc_pvals(self, cutcounts_file, pvals_outpath, fit_res_path) -> ProcessorOutputData:
        agg_cutcounts = self.extractor.extract_mappable_agg_cutcounts(
            cutcounts_file,
            self.gp.mappable_bases_file
        )

        self.gp.logger.debug(f'{self.chrom_name}: Estimating proportion of background signal')
        min_signal_quantile = (agg_cutcounts > 4).sum() / agg_cutcounts.count()
        if min_signal_quantile < 0.02:
            self.gp.logger.warning(f"{self.chrom_name}: Not enough signal to fit the background model. {min_signal_quantile*100:.2f}% (<2%) of data have # of cutcounts more than 4.")
            raise NotEnoughDataForContig
        
        # Step with window to speed it up
        s_fit = SegmentFit(self.genomic_interval, self.config, logger=self.gp.logger)
        per_window_trs_global, rmseas, global_fit = s_fit.fit_segment_thresholds(
            agg_cutcounts,
            step=self.config.signal_prop_sampling_step
        )
        
        # Various checks
        if global_fit.rmsea > self.config.rmsea_tr:
            self.gp.logger.warning(f"{self.chrom_name}: Not enough data to fit the background model. Best RMSEA: {global_fit.rmsea:.3f}. Chromosome fit might be poorly approximated.")

        self.gp.logger.debug(f"{self.chrom_name}: Signal quantile: {global_fit.fit_quantile:.3f}. signal threshold: {global_fit.fit_threshold:.0f}. Best RMSEA: {global_fit.rmsea:.3f}")

        good_fits_n = np.sum(rmseas <= self.config.rmsea_tr)
        n_rmsea = np.sum(~np.isnan(rmseas))
        self.gp.logger.debug(f"{self.chrom_name}: Signal thresholds approximated. {good_fits_n:,}/{n_rmsea:,} strided windows have RMSEA <= {self.config.rmsea_tr:.2f}")

        # Segmentation
        seg = BabachiWrapper(self.gp.logger, self.config)
        bad_segments = seg.run_segmentation(
            agg_cutcounts,
            per_window_trs_global,
            global_fit,
            self.chrom_name,
            self.chrom_size
        )

        self.gp.logger.debug(
            f'{self.chrom_name}: Estimating per-bp parameters of background model for {len(bad_segments)} segments'
        )

        chrom_fit = ChromosomeFit(self.genomic_interval, self.config, logger=self.gp.logger)
        fit_res, per_window_trs, final_rmsea, per_interval_params = chrom_fit.fit_params(
            agg_cutcounts=agg_cutcounts,
            bad_segments=bad_segments,
            global_fit=global_fit
        )

        outdir = pvals_outpath.replace('.pvals.parquet', '.fit_results.parquet')
        df = pd.DataFrame({
            'sliding_r': fit_res.r,
            'sliding_p': fit_res.p,
            'tr': per_window_trs,
            'inital_tr': per_window_trs_global,
            'bad': seg.annotate_with_segments(agg_cutcounts.shape, bad_segments),
            'rmsea': final_rmsea,
            'enough_bg': fit_res.enough_bg_mask
        })
        self.to_parquet(df, fit_res_path, compression_level=0)
        del df, per_window_trs, final_rmsea, bad_segments
        gc.collect()
  
        self.gp.logger.debug(f'{self.chrom_name}: Calculating p-values')

        pval_estimator = PvalueEstimator(self.config, self.gp.logger, name=self.chrom_name)

        # Strip masks to free up some memory
        agg_cutcounts = np.floor(agg_cutcounts.filled(np.nan))
        neglog_pvals = pval_estimator.estimate_pvalues(agg_cutcounts, fit_res)
        self.gp.logger.debug(f"Saving p-values for {self.chrom_name}")
        fname = f"{outdir}.{self.chrom_name}.inf_pvals.parquet"
        neglog_pvals = pd.DataFrame.from_dict({'log10_pval': fix_inf_pvals(neglog_pvals, fname)})
        self.to_parquet(neglog_pvals, pvals_outpath)

        return ProcessorOutputData(self.chrom_name, per_interval_params)

    @ensure_contig_exists
    def modwt_smooth_density(self, cutcounts_path, total_cutcounts, save_path):
        """
        Run MODWT smoothing on cutcounts.
        """
        filter = 'haar'
        self.gp.logger.debug(f"{self.chrom_name}: Running modwt signal smoothing (filter={filter}, level={self.config.modwt_level})")
        agg_cutcounts = self.extractor.extract_aggregated_cutcounts(cutcounts_path)
        smoothed = modwt_smooth(agg_cutcounts, filter, level=self.config.modwt_level)
        data = pd.DataFrame({
            'smoothed': smoothed,
            'normalized_density': normalize_density(agg_cutcounts, total_cutcounts) 
        })
        self.to_parquet(data, save_path)


    @ensure_contig_exists
    def call_hotspots(self, fdr_path, fdr_threshold=0.05) -> ProcessorOutputData:
        log10_fdrs = self.extractor.extract_fdr_track(fdr_path)
        self.gp.logger.debug(f"{self.chrom_name}: Calling hotspots")
        signif_fdrs = (log10_fdrs >= -np.log10(fdr_threshold)).astype(np.float32)
        bn_wrapper = BottleneckWrapper(self.config)
        smoothed_signif = bn_wrapper.centered_running_nansum(
            signif_fdrs,
            window=self.config.window,
        ) > 0
        region_starts, region_ends = find_stretches(smoothed_signif)

        max_log10_fdrs = np.empty(region_ends.shape)
        for i in range(len(region_starts)):
            start = region_starts[i]
            end = region_ends[i]
        
            max_log10_fdrs[i] = np.nanmax(log10_fdrs[start:end])
        
        self.gp.logger.debug(f"{self.chrom_name}: {len(region_starts)} hotspots called")

        data = pd.DataFrame({
            'start': region_starts,
            'end': region_ends,
            'max_neglog10_fdr': max_log10_fdrs
        })
        return ProcessorOutputData(self.chrom_name, data)
    

    @ensure_contig_exists
    def call_variable_width_peaks(self, smoothed_signal_path, fdr_path, fdr_threshold) -> ProcessorOutputData:
        signal_df = self.extractor.extract_from_parquet(
            smoothed_signal_path,
            columns=['smoothed', 'normalized_density']
        )

        signif_fdrs = self.extractor.extract_fdr_track(fdr_path) >= -np.log10(fdr_threshold)
        starts, ends = find_stretches(signif_fdrs)
        if len(starts) == 0:
            self.gp.logger.warning(f"{self.chrom_name}: No peaks found. Skipping...")
            raise NotEnoughDataForContig

        normalized_density = signal_df['normalized_density'].values
        self.gp.logger.debug(f"{self.chrom_name}: Calling peaks")
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
        self.gp.logger.debug(f"{self.chrom_name}: {len(peaks_df)} peaks called")

        return ProcessorOutputData(self.chrom_name, peaks_df)

    
    @ensure_contig_exists
    def to_parquet(self, data_df, path, compression_level=22):
        """
        Workaround for writing parquet files for chromosomes in parallel.
        """
        if data_df is None:
            raise NotEnoughDataForContig
        if isinstance(data_df, ProcessorOutputData):
            data_df = data_df.data_df
        data_df['chrom'] = pd.Categorical(
            [self.chrom_name] * data_df.shape[0],
            categories=[x for x in self.gp.chrom_sizes.keys()]
        )
        os.makedirs(path, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.gp.tmp_dir) as temp_dir:
            temp_path = os.path.join(temp_dir, f'{self.chrom_name}.temp.parquet')
            to_parquet_high_compression(data_df, temp_path, compression_level=compression_level)
            res_path = os.path.join(path, f'chrom={self.chrom_name}')
            if os.path.exists(res_path):
                shutil.rmtree(res_path)
            shutil.move(os.path.join(temp_path, f'chrom={self.chrom_name}'), path)
        
    
    @ensure_contig_exists
    def total_cutcounts(self, cutcounts):
        self.gp.logger.debug(f"{self.chrom_name}: Calculating total cutcounts")
        return self.extractor.extract_cutcounts(cutcounts).sum()


    @ensure_contig_exists
    def extract_density(self, smoothed_signal) -> ProcessorOutputData:
        data_df = self.extractor.extract_from_parquet(
            smoothed_signal, 
            columns=['chrom', 'normalized_density']
        )[::self.config.density_step]
        
        data_df['start'] = np.arange(len(data_df)) * self.config.density_step
        data_df.query('normalized_density > 0', inplace=True)
        return ProcessorOutputData(self.chrom_name, data_df)
