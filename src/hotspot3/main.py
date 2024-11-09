import logging
import argparse
import numpy as np
import os
from genome_tools.helpers import df_to_tabix

from hotspot3.processors import GenomeProcessor
from hotspot3.io import read_chrom_sizes
from hotspot3.io.logging import setup_logger
from hotspot3.config import ProcessorConfig


def main() -> None:
    args, logger_level = parse_arguments()
    root_logger = setup_logger()
    chrom_sizes = read_chrom_sizes(args.chrom_sizes)
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
        chrom_sizes=chrom_sizes,
        mappable_bases_file=args.mappable_bases,
        chromosomes=args.chromosomes,
        config=config,
    )
    precomp_pvals = args.pvals_parquet
    cutcounts_path = args.cutcounts
    smoothed_signal_path = args.signal_parquet
    sample_id = args.id

    debug_dir = f"{args.outdir}/debug/"
    os.makedirs(debug_dir, exist_ok=True)

    main_dir_prefix = f"{args.outdir}/{sample_id}"
    debug_dir_prefix = f"{args.outdir}/debug/{sample_id}"
    

    if smoothed_signal_path is None or precomp_pvals is None:
        if cutcounts_path is None:
            cutcounts_path = f"{main_dir_prefix}.cutcounts.bed.gz"
            genome_processor.write_cutcounts(args.bam, cutcounts_path)
            total_cutcounts = genome_processor.get_total_cutcounts(cutcounts_path)
            np.savetxt(f"{main_dir_prefix}.total_cutcounts", [total_cutcounts], fmt='%d')
        
        if smoothed_signal_path is None:
            smoothed_signal_path = f"{debug_dir_prefix}.smoothed_signal.parquet"
            genome_processor.smooth_signal_modwt(
                cutcounts_path,
                total_cutcounts=total_cutcounts,
                save_path=smoothed_signal_path
            )

        if precomp_pvals is None:
            fit_params_path = f"{debug_dir_prefix}.smoothed_signal.parquet"
            per_region_stats = genome_processor.fit_background_model(
                cutcounts_path,
                fit_params_path
            ).data_df
            per_region_stats.to_csv(f"{debug_dir_prefix}.fit_stats.tsv.gz", sep='\t', index=False)

            precomp_pvals = f"{main_dir_prefix}.pvals.parquet"
            genome_processor.calc_pvals(
                cutcounts_path,
                fit_params_path,
                precomp_pvals,
            )
            
    precomp_fdrs = f"{debug_dir_prefix}.fdrs.parquet"
    genome_processor.calc_fdr(precomp_pvals, precomp_fdrs, max(args.fdrs))

    root_logger.info(f'Calling peaks and hotspots at FDRs: {args.fdrs}') 
    for fdr in args.fdrs:
        fdr_pref = f"{args.outdir}/fdr{fdr}/{sample_id}"
        os.makedirs(f"{args.outdir}/fdr{fdr}", exist_ok=True)
        root_logger.debug(f'Calling hotspots at FDR={fdr}')
        hotspots = genome_processor.call_hotspots(
            precomp_fdrs,
            sample_id,
            fdr_tr=fdr
        ).data_df[['chrom', 'start', 'end', 'id', 'score', 'max_neglog10_fdr']]
        hotspots_path = f"{fdr_pref}.hotspots.fdr{fdr}.bed.gz"
        df_to_tabix(hotspots, hotspots_path)
        # df_to_bigbed

        root_logger.debug(f'Calling variable width peaks at FDR={fdr}')
        peaks = genome_processor.call_variable_width_peaks(
            smoothed_signal_path,
            fdrs_path=precomp_fdrs,
            fdr_tr=fdr
        ).data_df
        peaks['id'] = sample_id
        peaks = peaks[['chrom', 'start', 'end', 'id', 'max_density', 'summit']]
        peaks_path = f"{fdr_pref}.peaks.fdr{fdr}.bed.gz"
        df_to_tabix(peaks, peaks_path)
        # df_to_bigbed

    if args.save_density:
        root_logger.info('Saving density')
        denisty_path = f"{main_dir_prefix}.normalized_density.bw"
        genome_processor.extract_normalized_density(smoothed_signal_path, denisty_path)
    root_logger.info('Program finished')


def parse_arguments(extra_desc: str = "") -> argparse.Namespace:
    name = "Run hotspot2 peak calling" + extra_desc
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
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use", default=1)
    parser.add_argument("--outdir", help="Path to output directory", default=".")
    parser.add_argument("--debug", help="Run in debug mode. Adds additional prints and saves background window fit params", action='store_true', default=False)
    parser.add_argument("--tempdir", help="Path to temporary directory. Defaults to system temp directory", default=None)

    # Arguments for calculating p-values
    parser.add_argument("--mappable_bases", help="Path to mappable bases file (if needed). Used in fit of background model", default=None)
    parser.add_argument("--window", help="Window size for smoothing cutcounts", type=int, default=151)
    parser.add_argument("--background_window", help="Background window size", type=int, default=50001)
    parser.add_argument("--signal_quantile", help="Chromosome signal quantile. Positions with signal above the threshold considered to be 'potential peaks' and are not used to fit a background model. Ignored if --adaptive_signal_tr is provided", type=float, default=0.99)
    
    # Arguments to skip previous steps if provided
    parser.add_argument("--bam", help="Path to input bam/cram file", default=None)
    parser.add_argument("--cutcounts", help="Path to pre-calculated cutcounts tabix file. Skip extracting cutcounts from bam file", default=None)
    parser.add_argument("--signal_parquet", help="Path to pre-calculated partitioned parquet file(s) with per-bp smoothed signal. Skips modwt signal smoothing", default=None)
    parser.add_argument("--pvals_parquet", help="Path to pre-calculated partitioned parquet file(s) with per-bp p-values. Skips p-value calculation", default=None)

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
