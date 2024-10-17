import logging
import argparse
from genome_tools.helpers import df_to_tabix

from hotspot3.processors import GenomeProcessor, root_logger, set_logger_config
from hotspot3.utils import read_chrom_sizes


def main() -> None:
    args, logger_level = parse_arguments()
    chrom_sizes = read_chrom_sizes(args.chrom_sizes)
    genome_processor = GenomeProcessor(
        chrom_sizes=chrom_sizes,
        mappable_bases_file=args.mappable_bases,
        tmp_dir=args.tempdir,
        cpus=args.cpus,
        logger_level=logger_level,
        save_debug=args.debug,
        bg_window=args.background_window,
        window=args.window,
        min_hotspot_width=args.min_hotspot_width,
        #chromosomes=['chr3']
    )
    precomp_fdrs = args.fdrs_parquet
    cutcounts_path = args.cutcounts
    smoothed_signal_path = args.signal_parquet
    sample_id = args.id
    outdir_pref = f"{args.outdir}/{sample_id}"

    if smoothed_signal_path is None or precomp_fdrs is None:
        if cutcounts_path is None:
            cutcounts_path = f"{outdir_pref}.cutcounts.bed.gz"
            genome_processor.write_cutcounts(args.bam, cutcounts_path)

        if smoothed_signal_path is None:
            smoothed_signal_path = f"{outdir_pref}.smoothed_signal.parquet"
            genome_processor.modwt_smooth_signal(cutcounts_path, smoothed_signal_path)

        if precomp_fdrs is None:
            precomp_pvals = f"{outdir_pref}.pvals.parquet"
            genome_processor.calc_pval(cutcounts_path, precomp_pvals)
    
            precomp_fdrs = f"{outdir_pref}.fdrs.parquet"
            genome_processor.calc_fdr(precomp_pvals, precomp_fdrs)

    root_logger.info(f'Calling peaks and hotspots at FDRs: {args.fdrs}') 
    for fdr in args.fdrs:
        root_logger.debug(f'Calling hotspots at FDR={fdr}')
        hotspots = genome_processor.call_hotspots(
            precomp_fdrs,
            sample_id,
            fdr_tr=fdr
        ).data_df[['chrom', 'start', 'end', 'id', 'score', 'max_neglog10_fdr']]
        hotspots_path = f"{outdir_pref}.hotspots.fdr{fdr}.bed.gz"
        df_to_tabix(hotspots, hotspots_path)

        root_logger.debug(f'Calling variable width peaks at FDR={fdr}')
        peaks = genome_processor.call_variable_width_peaks(
            smoothed_signal_path,
            fdrs_path=precomp_fdrs,
            fdr_tr=fdr
        ).data_df
        peaks['id'] = sample_id
        peaks = peaks[['chrom', 'start', 'end', 'id', 'max_density', 'summit']]
        peaks_path = f"{outdir_pref}.peaks.fdr{fdr}.bed.gz"
        df_to_tabix(peaks, peaks_path)

    if args.save_density:
        root_logger.info('Saving density')
        density_data = genome_processor.extract_density(smoothed_signal_path).data_df
        density_data = density_data[['chrom', 'start', 'end', 'id', 'normalized_density']]
        denisty_path = f"{outdir_pref}.density.bed.gz"
        df_to_tabix(density_data, denisty_path)
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
        - tabix indexed normalized density of cutcounts: {sample_id}.density.bed.gz
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

    parser.add_argument("--min_hotspot_width", help="Minimum width for a region to be called a hotspot", type=int, default=50)
    
    # Arguments to skip previous steps if provided
    parser.add_argument("--bam", help="Path to input bam/cram file", default=None)
    parser.add_argument("--cutcounts", help="Path to pre-calculated cutcounts tabix file. Skip extracting cutcounts from bam file", default=None)
    parser.add_argument("--signal_parquet", help="Path to pre-calculated partitioned parquet file(s) with per-bp smoothed signal. Skips modwt signal smoothing", default=None)
    parser.add_argument("--fdrs_parquet", help="Path to pre-calculated partitioned parquet file(s) with per-bp FDRs. Skips p-value calculation", default=None)

    # Optional - save density
    parser.add_argument("--save_density", action='store_true', help="Save density of cutcounts", default=False)

    args = parser.parse_args()
    logger_level = logging.DEBUG if args.debug else logging.INFO
    set_logger_config(root_logger, logger_level)

    
    if args.signal_parquet is not None and args.fdrs_parquet is not None:
        ignored_atrs = ['cutcounts', 'bam', 'mappable_bases']
        for atr in ignored_atrs:
            if getattr(args, atr) is not None:
                root_logger.warning(f"Ignoring {atr}. Precomputed smoothed signal and per-bp FDRs are provided")
    
    elif args.cutcounts is not None and args.bam is not None:
        root_logger.warning("Ignoring bam file. Precomputed cutcounts are provided")
    elif args.cutcounts is None and args.bam is None:
        parser.error("Either provide both precomputed FDRs and smoothed signal or bam/cutcounts")
    return args, logger_level


if __name__ == "__main__":
    main()
