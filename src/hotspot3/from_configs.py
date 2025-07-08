import os
import shutil

from hotspot3.processors import GenomeProcessor
from hotspot3.io.paths import Hotspot3Paths


def run_from_configs(
        genome_processor: GenomeProcessor, 
        paths: Hotspot3Paths, 
        fdrs,
    ):
    step_names = paths.find_missing_steps()
    genome_processor.logger.info(f"Running: {', '.join(paths.get_display_names(step_names))}")
    if 'cutcounts' in step_names:
        genome_processor.extract_cutcounts_from_bam(paths.bam, paths.cutcounts)
    
    if 'total_cutcounts' in step_names:
        genome_processor.get_total_cutcounts(paths.cutcounts, paths.total_cutcounts)
    
    if 'smoothed_signal' in step_names:
        genome_processor.smooth_signal_modwt(
            paths.cutcounts,
            save_path=paths.smoothed_signal,
            total_cutcounts_path=paths.total_cutcounts
        )
    if 'fit_params' in step_names:
        genome_processor.fit_background_model(
            paths.cutcounts,
            total_cutcounts_path=paths.total_cutcounts,
            save_path=paths.fit_params,
            per_region_stats_path=paths.per_region_stats,
            per_region_stats_path_bw=paths.per_region_background,
        )

        genome_processor.extract_fit_thresholds_to_bw(
            paths.fit_params,
            paths.total_cutcounts,
            paths.thresholds,
        )

        genome_processor.extract_bg_quantile_to_bw(
            paths.fit_params,
            paths.total_cutcounts,
            paths.background
        )
    
    if 'pvals' in step_names:
        genome_processor.calc_pvals(
            paths.cutcounts,
            paths.fit_params,
            paths.pvals
        )
    
    if 'fdrs' in step_names:
        genome_processor.calc_fdr(
            paths.pvals,
            paths.fdrs,
            max(fdrs)
        )
    
    if 'normalized_density' in step_names:
        genome_processor.extract_normalized_density(
            paths.smoothed_signal,
            paths.normalized_density
        )
    
    if 'peak_calling' in step_names:
        genome_processor.logger.info(f'Calling peaks and hotspots at FDRs: {fdrs}') 
        for fdr in fdrs:
            fdr_dir = paths.fdrs_dir(fdr)
            if os.path.exists(fdr_dir):
                shutil.rmtree(fdr_dir)
            os.makedirs(fdr_dir, exist_ok=True)

            genome_processor.logger.debug(f'Calling hotspots at FDR={fdr}')
            genome_processor.call_hotspots(
                paths.fdrs,
                sample_id=genome_processor.sample_id,
                save_path=paths.hotspots(fdr),
                save_path_bb=paths.hotspots_bb(fdr),
                fdr_tr=fdr
            )

            genome_processor.logger.debug(f'Calling variable width peaks at FDR={fdr}')
            genome_processor.call_variable_width_peaks(
                paths.smoothed_signal,
                paths.fdrs,
                paths.total_cutcounts,
                sample_id=genome_processor.sample_id,
                save_path=paths.peaks(fdr),
                save_path_bb=paths.peaks_bb(fdr),
                fdr_tr=fdr
            )
