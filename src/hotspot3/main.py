import logging
import argparse
import os
import shutil

from hotspot3.processors import GenomeProcessor
from hotspot3.io.logging import setup_logger
from hotspot3.config import ProcessorConfig


def main() -> None:
    args, logger_level = parse_arguments()
    root_logger = setup_logger()
    config = ProcessorConfig(
        window=args.window,
        bg_window=args.background_window,
        max_background_prop=args.signal_quantile,
        save_debug=args.debug,
        cpus=args.cpus,
        logger_level=logger_level,
        tmp_dir=args.tempdir,
    )
    genome_processor = GenomeProcessor(
        sample_id=args.id,
        chrom_sizes_file=args.chrom_sizes,
        mappable_bases_file=args.mappable_bases,
        chromosomes=args.chromosomes,
        reference_fasta=args.reference,
        config=config,
    )
    precomp_pvals = args.pvals_parquet
    cutcounts_path = args.cutcounts
    smoothed_signal_path = args.signal_parquet
    precomp_fdrs = args.fdrs_parquet
    sample_id = args.id

    debug_dir = f"{args.outdir}/debug/"
    os.makedirs(debug_dir, exist_ok=True)

    main_dir_prefix = f"{args.outdir}/{sample_id}"
    debug_dir_prefix = f"{args.outdir}/debug/{sample_id}"
    
    if cutcounts_path is None:
        total_cutcounts_path = None
    else:
        total_cutcounts_path = cutcounts_path.replace('.cutcounts.bed.gz', '.total_cutcounts')

    if smoothed_signal_path is None or precomp_pvals is None:
        if cutcounts_path is None:
            cutcounts_path = f"{main_dir_prefix}.cutcounts.bed.gz"
            total_cutcounts_path = f"{main_dir_prefix}.total_cutcounts"
            genome_processor.extract_cutcounts_from_bam(args.bam, cutcounts_path)

            genome_processor.get_total_cutcounts(cutcounts_path, total_cutcounts_path)
            
        
        if smoothed_signal_path is None:
            smoothed_signal_path = f"{debug_dir_prefix}.smoothed_signal.parquet"
            
            genome_processor.smooth_signal_modwt(
                cutcounts_path,
                save_path=smoothed_signal_path,
                total_cutcounts_path=total_cutcounts_path
            )

        if precomp_pvals is None and precomp_fdrs is None:
            fit_params_path = f"{debug_dir_prefix}.fit_params.parquet"

            per_region_stats_path = f"{main_dir_prefix}.fit_stats.tsv.gz"
            per_region_stats_path_bw = f"{main_dir_prefix}.per_segment.background.bw"
            
            threholds_bw_path = f"{main_dir_prefix}.threholds.bw"
            background_bw_path = f"{main_dir_prefix}.background.bw"
            genome_processor.fit_background_model(
                cutcounts_path,
                total_cutcounts_path=total_cutcounts_path,
                save_path=fit_params_path,
                per_region_stats_path=per_region_stats_path,
                per_region_stats_path_bw=per_region_stats_path_bw,
            )

            precomp_pvals = f"{main_dir_prefix}.pvals.parquet"
            genome_processor.calc_pvals(
                cutcounts_path,
                fit_params_path,
                precomp_pvals,
            )

            genome_processor.extract_thresholds_to_bw(
                fit_params_path,
                total_cutcounts_path,
                threholds_bw_path,
            )

            genome_processor.extract_bg_quantile_to_bw(
                fit_params_path,
                total_cutcounts_path,
                background_bw_path
            )
    
    if precomp_fdrs is None:
        precomp_fdrs = f"{debug_dir_prefix}.fdrs.parquet"
        genome_processor.calc_fdr(precomp_pvals, precomp_fdrs, max(args.fdrs))

    root_logger.info(f'Calling peaks and hotspots at FDRs: {args.fdrs}') 
    for fdr in args.fdrs:
        fdr_dir = f"{args.outdir}/fdr{fdr}"
        if os.path.exists(fdr_dir):
            shutil.rmtree(fdr_dir)
        os.makedirs(fdr_dir, exist_ok=True)
        fdr_pref = f"{fdr_dir}/{sample_id}"
        
        hotspots_path = f"{fdr_pref}.hotspots.fdr{fdr}.bed.gz"
        hotspots_path_bb = f"{fdr_pref}.hotspots.fdr{fdr}.bb"
        root_logger.debug(f'Calling hotspots at FDR={fdr}')
        genome_processor.call_hotspots(
            precomp_fdrs,
            sample_id=sample_id,
            save_path=hotspots_path,
            save_path_bb=hotspots_path_bb,
            fdr_tr=fdr
        )

        root_logger.debug(f'Calling variable width peaks at FDR={fdr}')
        peaks_path = f"{fdr_pref}.peaks.fdr{fdr}.bed.gz"
        peaks_path_bb = f"{fdr_pref}.peaks.fdr{fdr}.bb"
        genome_processor.call_variable_width_peaks(
            smoothed_signal_path,
            precomp_fdrs,
            total_cutcounts_path,
            sample_id=sample_id,
            save_path=peaks_path,
            save_path_bb=peaks_path_bb,
            fdr_tr=fdr
        )

    if args.save_density:
        root_logger.info('Saving density')
        denisty_path = f"{main_dir_prefix}.normalized_density.bw"
        genome_processor.extract_normalized_density(smoothed_signal_path, denisty_path)
    root_logger.info('Program finished')


def parse_arguments(extra_desc: str = "") -> argparse.Namespace:
    name = "Run hotspot3 peak calling" + extra_desc
    # TODO: Change description to reflect the new functionality
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=name + """
    
    Saves the following temporary files in the output directory:
        - tabix indexed cutcounts: {sample_id}.cutcounts.bed.gz (~200MB)
        - per-bp smoothed signal: {sample_id}.smoothed_signal.parquet (large, ~10GB)
        - per-bp raw p-values: {sample_id}.pvals.parquet (large, ~1.5GB)
        - parameters used for background per-chromosome fits: {sample_id}.pvals.params.parquet (~1.5MB)
        - per-bp FDR estimates: {sample_id}.fdrs.parquet (~600MB)
    
    Note:
        Multiple FDR thresholds can be provided as a space-separated list.

    For each FDR threshold:
        - tabix indexed hotspots at FDR: {sample_id}.hotspots.fdr{fdr}.bed.gz
        - tabix indexed peaks at FDR: {sample_id}.peaks.fdr{fdr}.bed.gz
    
    To quickly find hotspots and peaks at other FDRs than initially provided, specify 
    --fdrs_parquet {sample_id}.fdrs.parquet 
    and 
    --signal_parquet {sample_id}.smoothed_signal.parquet.

    Or --fdrs_parquet {sample_id}.fdrs.parquet 
    and
    --cutcounts {sample_id}.cutcounts.bed.gz or --bam input.bam
    if signal_parquet was deleted (will take more time smoothing the signal).
    
    Optional if --save_density is provided:
        - normalized density of cutcounts in bigwig format: {sample_id}.normalized_density.bw
    """
    )
    
    # common arguments
    parser.add_argument("id", type=str, help="Unique identifier of the sample")

    parser.add_argument("--chrom_sizes", help="Path to chromosome sizes file. If none assumed to be hg38 sizes", default=None)
    parser.add_argument(
        "--fdrs", help="List of FDR thresholds, space separated", type=float, 
        nargs='+', default=[0.05]
    )
    parser.add_argument("--reference", help="Path to reference fasta file. Required to work with cram files with missing fasta", default=None)
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use", default=1)
    parser.add_argument("--outdir", help="Path to output directory", default=".")
    parser.add_argument("--debug", help="Run in debug mode. Adds additional prints and saves background window fit params", action='store_true', default=False)
    parser.add_argument("--tempdir", help="Path to temporary directory. Defaults to system temp directory", default=None)

    # Arguments for calculating p-values
    parser.add_argument("--mappable_bases", help="Path to mappable bases file (if needed). Used in fit of background model", default=None)
    parser.add_argument("--window", help="Window size for smoothing cutcounts", type=int, default=151)
    parser.add_argument("--background_window", help="Background window size", type=int, default=50001)
    parser.add_argument(
        "--signal_quantile",
        help="Chromosome signal quantile. Positions with signal above the threshold considered to be 'potential peaks' and are not used to fit a background model",
        type=float,
        default=0.995
    )
    
    # Arguments to skip previous steps if provided
    parser.add_argument("--bam", help="Path to input bam/cram file", default=None)
    parser.add_argument("--cutcounts", help="Path to pre-calculated cutcounts tabix file. Skip extracting cutcounts from bam file", default=None)
    parser.add_argument("--signal_parquet", help="Path to pre-calculated partitioned parquet file(s) with per-bp smoothed signal. Skips modwt signal smoothing", default=None)
    parser.add_argument("--pvals_parquet", help="Path to pre-calculated partitioned parquet file(s) with per-bp p-values. Skips p-value calculation", default=None)
    parser.add_argument("--fdrs_parquet", help="Path to pre-calculated fdrs. Can correct for several sampels using multiple_samples_fdr.py", default=None)

    parser.add_argument("--chromosomes", help="List of chromosomes to process. Used for debug", nargs='+', default=None)

    # Optional - save density
    parser.add_argument("--save_density", action='store_true', help="Save density of cutcounts", default=False)

    args = parser.parse_args()
    logger_level = logging.DEBUG if args.debug else logging.INFO
    root_logger = setup_logger(level=logger_level)

    
    if args.signal_parquet is not None and args.pvals_parquet is not None:
        ignored_atrs = ['cutcounts', 'bam', 'mappable_bases']
        for atr in ignored_atrs:
            if getattr(args, atr) is not None:
                root_logger.warning(f"Ignoring {atr}. Precomputed smoothed signal and per-bp P-values are provided")
    
    elif args.cutcounts is not None and args.bam is not None:
        root_logger.warning("Ignoring bam file. Precomputed cutcounts are provided")
    elif args.cutcounts is None and args.bam is None:
        parser.error("Either provide both precomputed P-values and smoothed signal or bam/cutcounts")
    
    return args, logger_level


if __name__ == "__main__":
    main()
