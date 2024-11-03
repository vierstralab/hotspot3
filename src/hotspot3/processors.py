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
import dataclasses
from hotspot3.logging import setup_logger
from hotspot3.models import ProcessorOutputData, NoContigPresentError, ProcessorConfig
from hotspot3.file_extractors import ChromosomeExtractor
from hotspot3.pvalue import PvalueEstimator
from hotspot3.fit import GlobalBackgroundFit, WindowBackgroundFit, StridedFit

from hotspot3.signal_smoothing import modwt_smooth
from hotspot3.peak_calling import find_stretches, find_varwidth_peaks
from hotspot3.stats import logfdr_from_logpvals, fix_inf_pvals
from hotspot3.utils import normalize_density, is_iterable, to_parquet_high_compression, delete_path, df_to_bigwig, ensure_contig_exists, interpolate_nan


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
    Main class to run hotspot2-related functions and store parameters.

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

    def calc_pval(self, cutcounts_file, pvals_path: str):
        self.logger.info('Calculating per-bp p-values')
        delete_path(pvals_path)
        self.parallel_by_chromosome(
            ChromosomeProcessor.calc_pvals,
            cutcounts_file,
            pvals_path
        )
    
    def calc_fdr(self, pvals_path, fdrs_path, max_fdr=1):
        self.logger.info('Calculating per-bp FDRs')
        log_pval = pd.read_parquet(
            pvals_path,
            engine='pyarrow', 
            columns=['chrom', 'log10_pval']
        ) 
        # file is always sorted within chromosomes
        chrom_pos_mapping = log_pval['chrom'].drop_duplicates()
        starts = chrom_pos_mapping.index
        ends = [*starts[1:], log_pval.shape[0]]
        log_pval = log_pval['log10_pval'].values

        result = np.full_like(log_pval, np.nan)
        mask = ~np.isnan(log_pval)
        not_nan_shape = np.sum(mask)
        mask = mask & (log_pval >= -np.log10(max_fdr))

        # Cast to natural log
        log_pval = log_pval[mask]
        log_pval *= -np.log(10).astype(log_pval.dtype)

        result[mask] = logfdr_from_logpvals(log_pval, method=self.config.fdr_method, m=not_nan_shape)
        del log_pval
        gc.collect()
        # Cast to neglog10
        result /= -np.log(10).astype(result.dtype)

        result = [
            ProcessorOutputData(
                chrom, 
                pd.DataFrame({'log10_fdr': result[start:end]})
            )
            for chrom, start, end
            in zip(chrom_pos_mapping, starts, ends)
        ]
        gc.collect()
        delete_path(fdrs_path)
        self.logger.debug('Saving per-bp FDRs')
        self.parallel_by_chromosome(
            ChromosomeProcessor.to_parquet,
            result,
            fdrs_path,
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
        self.chrom_size = self.gp.chrom_sizes[chrom_name]
        self.extractor = ChromosomeExtractor(chrom_name, self.chrom_size, config=self.config)

    @ensure_contig_exists
    def calc_pvals(self, cutcounts_file, pvals_outpath) -> ProcessorOutputData:
        self.gp.logger.debug(f'Aggregating cutcounts for chromosome {self.chrom_name}')
        
        agg_cutcounts = self.extractor.extract_mappable_agg_cutcounts(
            cutcounts_file,
            self.gp.mappable_bases_file
        )

        self.gp.logger.debug(
            f"Cutcounts aggregated for {self.chrom_name}, {agg_cutcounts.count():,}/{agg_cutcounts.shape[0]:,} bases are mappable"
        )

        self.gp.logger.debug(f'Estimating background signal for {self.chrom_name}')
        g_fit = GlobalBackgroundFit(self.config)
        global_fit = g_fit.fit(agg_cutcounts)
        global_p = global_fit.p
        global_r = global_fit.r
        # params_df = pd.DataFrame({
        #     'unique_cutcounts': unique_cutcounts,
        #     'count': n_obs,
        #     'outliers_tr': [outliers_tr] * len(unique_cutcounts),
        #     'mean': [global_fit.mean] * len(unique_cutcounts),
        #     'variance': [global_fit.var] * len(unique_cutcounts),
        #     'rmsea': [global_fit.rmsea] * len(unique_cutcounts),
        #     'epsilon': [epsilon_global] * len(unique_cutcounts),
        #     'epsilon_mu': [epsilon_mu_global] * len(unique_cutcounts),
        # })
        # self.gp.logger.debug(f"Writing pvals for {self.chrom_name}")
        # self.to_parquet(params_df, params_outpath)
        self.gp.logger.debug(f"Global fit finished for {self.chrom_name}.")
        self.gp.logger.debug(f"{self.chrom_name}: signal quantile: {global_fit.fit_quantile:.3f}. signal threshold: {global_fit.fit_threshold:.0f}. Best RMSEA: {global_fit.rmsea:.3f}")
        

        self.gp.logger.debug(f"Estimating per-window signal thresholds for {self.chrom_name}")
        config = self.config # delete this line after testing
        rmsea_fit = StridedFit(config, name=self.chrom_name)
        per_window_trs, per_window_q, per_window_rmsea = rmsea_fit.fit_tr(agg_cutcounts, global_fit=global_fit)

        per_window_trs = interpolate_nan(per_window_trs)
        self.gp.logger.debug(f"Per-window signal thresholds calculated for {self.chrom_name}")

        w_fit = WindowBackgroundFit(self.config)
        self.gp.logger.debug(f"Estimating per-bp r and p for {self.chrom_name}")
        good_fit = interpolate_nan(per_window_rmsea) <= self.config.rmsea_tr # FIXME, don't interpolate rmsea
        fit_res = w_fit.fit(agg_cutcounts, per_window_trs=per_window_trs, where=good_fit)
        fit_res.r[~good_fit] = global_fit.r
        fit_res.p[~good_fit] = w_fit.fit(
            agg_cutcounts,
            per_window_trs=per_window_trs,
            where=~good_fit,
            global_r=global_r
        ).p[~good_fit]
        
        outdir = pvals_outpath.replace('.pvals.parquet', '')
        df = pd.DataFrame({
            'sliding_r': fit_res.r,
            'sliding_p': fit_res.p,
            'rmsea': per_window_rmsea,
            'tr': per_window_trs,
            'q': per_window_q,
        })
        fit_res_path = f"{outdir}.fit_results.parquet"
        self.to_parquet(df, fit_res_path)
        del df, per_window_q, per_window_rmsea, per_window_trs
        gc.collect()

        poisson_fits = fit_res.poisson_fit_params
        if poisson_fits is not None:
            self.gp.logger.debug(f"Using Poisson distribution for {len(poisson_fits):,}/{agg_cutcounts.count():,} windows for {self.chrom_name}")
  
        self.gp.logger.debug(f'Calculating p-values for {self.chrom_name}')

        pval_estimator = PvalueEstimator(self.config, self.gp.logger, name=self.chrom_name)

        # Strip masks to free up some memory
        agg_cutcounts = np.floor(agg_cutcounts.filled(np.nan))

        neglog_pvals = pval_estimator.estimate_pvalues(
            agg_cutcounts,
            fit_res.r,
            fit_res.p,
            fit_res.enough_bg_mask,
            bad_fits=poisson_fits
        )
        del agg_cutcounts
        gc.collect()

        self.gp.logger.debug(f"Saving p-values for {self.chrom_name}")
        fname = f"{outdir}.{self.chrom_name}.inf_pvals.parquet"
        neglog_pvals = pd.DataFrame.from_dict({'log10_pval': fix_inf_pvals(neglog_pvals, fname)})
        self.to_parquet(neglog_pvals, pvals_outpath)
        del neglog_pvals
        gc.collect()

        # epsilon_global, epsilon_mu_global = calc_epsilon_and_epsilon_mu(global_r, global_p, tr=outliers_tr)

    @ensure_contig_exists
    def modwt_smooth_density(self, cutcounts_path, total_cutcounts, save_path):
        """
        Run MODWT smoothing on cutcounts.
        """
        agg_cutcounts = self.extractor.extract_aggregated_cutcounts(cutcounts_path)
        filters = 'haar'
        self.gp.logger.debug(f"Running modwt smoothing (filter={filters}, level={self.config.modwt_level}) for {self.chrom_name}")
        smoothed = modwt_smooth(agg_cutcounts, filters, level=self.config.modwt_level)
        data = pd.DataFrame({
            'smoothed': smoothed,
            'normalized_density': normalize_density(agg_cutcounts, total_cutcounts) 
        })
        self.to_parquet(data, save_path)


    @ensure_contig_exists
    def call_hotspots(self, fdr_path, fdr_threshold=0.05) -> ProcessorOutputData:
        log10_fdrs = self.extractor.extract_fdr_track(fdr_path)
        self.gp.logger.debug(f"Calling hotspots for {self.chrom_name}")
        signif_fdrs = (log10_fdrs >= -np.log10(fdr_threshold)).astype(np.float32)
        window_fit = WindowBackgroundFit(self.config)
        smoothed_signif = window_fit.centered_running_nansum(
            signif_fdrs,
            window=self.config.window,
            min_count=1,
        ) > 0
        region_starts, region_ends = find_stretches(smoothed_signif)

        max_log10_fdrs = np.empty(region_ends.shape)
        for i in range(len(region_starts)):
            start = region_starts[i]
            end = region_ends[i]
            if np.sum(~np.isnan(log10_fdrs[start:end])) == 0:
                print(log10_fdrs[start:end])
                print(start, end)
                print(smoothed_signif[start:end])
        
            max_log10_fdrs[i] = np.nanmax(log10_fdrs[start:end])
        
        self.gp.logger.debug(f"{len(region_starts)} hotspots called for {self.chrom_name}")

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
            self.gp.logger.warning(f"No peaks found for {self.chrom_name}. Skipping...")
            raise NoContigPresentError

        normalized_density = signal_df['normalized_density'].values
        self.gp.logger.debug(f"Calling peaks for {self.chrom_name}")
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
        self.gp.logger.debug(f"{len(peaks_df)} peaks called for {self.chrom_name}")

        return ProcessorOutputData(self.chrom_name, peaks_df)

    
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
            res_path = os.path.join(path, f'chrom={self.chrom_name}')
            if os.path.exists(res_path):
                shutil.rmtree(res_path)
            shutil.move(os.path.join(temp_path, f'chrom={self.chrom_name}'), path)
        
    
    @ensure_contig_exists
    def total_cutcounts(self, cutcounts):
        self.gp.logger.debug(f"Calculating total cutcounts for {self.chrom_name}")
        return self.extractor.extract_cutcounts(cutcounts).sum()


    @ensure_contig_exists
    def extract_density(self, smoothed_signal) -> ProcessorOutputData:
        data_df = self.extractor.extract_from_parquet(
            smoothed_signal, 
            columns=['chrom', 'normalized_density']
        )
        
        data_df['start'] = np.arange(len(data_df)) * self.config.density_step
        data_df.query('normalized_density > 0', inplace=True)
        return ProcessorOutputData(self.chrom_name, data_df)
