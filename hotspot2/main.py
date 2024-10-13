import logging
import gc
import argparse
from processors import GenomeProcessor, root_logger, set_logger_config
from utils import read_chrom_sizes, df_to_tabix, normalize_density

def main():
    args, logger_level = parse_arguments()
    chrom_sizes = read_chrom_sizes(args.chrom_sizes)
    genome_processor = GenomeProcessor(
        chrom_sizes=chrom_sizes,
        mappable_bases_file=args.mappable_bases,
        cpus=args.cpus,
        logger_level=logger_level,
        save_debug=args.debug,
        window=args.window,
        bg_window=args.background_window,
        #chromosomes=['chr20', 'chr19']
    )
    precomp_fdrs = args.precomp_fdrs
    cutcounts_path = args.cutcounts
    sample_id = args.id
    outdir_pref = f"{args.outdir}/{sample_id}"

    if precomp_fdrs is None:
        root_logger.info('Calculating p-values')
        pvals_data = genome_processor.calc_pval(cutcounts_path)
        root_logger.debug('Saving P-values')
        precomp_fdrs = f"{outdir_pref}.stats.parquet"
        pvals_data.data_df.to_parquet(
            precomp_fdrs,
            engine='pyarrow',
            compression='zstd',
            compression_level=22,
            index=False,
            partition_cols=['chrom'],
        )
        pvals_data.extra_df.to_csv(
            f"{outdir_pref}.params.gz",
            sep='\t',
            index=False
        )
        del pvals_data
        gc.collect()

    root_logger.info('Smoothing signal using MODWT')
    smoothed_data = genome_processor.modwt_smooth_signal(cutcounts_path)
    total_cutcounts = smoothed_data.extra_df['total_cutcounts'].sum()

    root_logger.info(f'Calling peaks and hotspots at FDRs: {args.fdrs}') 
    for fdr in args.fdrs:
        root_logger.debug(f'Calling hotspots at FDR={fdr}')
        hotspots = genome_processor.call_hotspots(
            precomp_fdrs,
            sample_id,
            fdr_tr=fdr
        ).data_df
        hotspots_path = f"{outdir_pref}.hotspots.fdr{fdr}.bed.gz"
        df_to_tabix(hotspots, hotspots_path)
        del hotspots
        gc.collect()

        root_logger.debug(f'Calling variable width peaks at FDR={fdr}')
        peaks = genome_processor.call_variable_width_peaks(
            smoothed_signal=smoothed_data,
            hotspots_path=hotspots_path,
        ).data_df
        peaks['id'] = sample_id
        peaks['max_density'] = normalize_density(
            peaks['max_density'], 
            total_cutcounts=total_cutcounts
        )
        peaks_path = f"{outdir_pref}.peaks.fdr{args.fdr}.bed.gz"
        df_to_tabix(peaks, peaks_path)
        del peaks
        gc.collect()

    if args.save_density:
        root_logger.info('Saving density')
        density_data = genome_processor.extract_density(smoothed_data).data_df
        density_data['density'] = normalize_density(
            density_data['density'],
            total_cutcounts=total_cutcounts
        )
        root_logger.debug('Saving densities')
        denisty_path = f"{outdir_pref}.density.bed.gz"
        df_to_tabix(density_data, denisty_path)
    root_logger.info('Program finished')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run hotspot2 to call hotspots and peaks from bam file")
    
    # common arguments
    parser.add_argument("id", type=str, help="Unique identifier of the sample")

    parser.add_argument("--chrom_sizes", help="Path to chromosome sizes file. If none assumed to be hg38 sizes", default=None)
    parser.add_argument(
        "--fdrs", help="List of FDR thresholds, comma separated", type=float, 
        nargs='+', default=[0.05]
    )
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use", default=1)
    parser.add_argument("--outdir", help="Path to output directory", default=".")
    parser.add_argument("--debug", help="Path to chromosome sizes file. If none assumed to be hg38 sizes", action='store_true', default=False)

    # Arguments for calculating p-values
    parser.add_argument("--mappable_bases", help="Path to mappable bases file (if needed)", default=None)
    parser.add_argument("--window", help="Window size for smoothing cutcounts", type=int, default=201)
    parser.add_argument("--background_window", help="Background window size", type=int, default=50001)
    
    # Arguments to skip previous steps if provided
    parser.add_argument("--bam", help="Path to input bam/cram file", default=None)
    parser.add_argument("--cutcounts", help="Path to pre-calculated cutcounts tabix file. Skip extracting cutcounts from bam file", default=None)
    parser.add_argument("--precomp_fdrs", help="Path to pre-calculated partitioned parquet file(s) with per-bp FDRs. Skips p-value calculation", default=None)

    # Optional - save density
    parser.add_argument("--save_density", action='store_true', help="Save density of cutcounts", default=False)

    args = parser.parse_args()
    logger_level = logging.DEBUG if args.debug else logging.INFO
    set_logger_config(root_logger, logger_level)

    if args.precomp_fdrs is not None:
        for arg_name in ['mappable_bases', 'precomp_cutcounts', 'bam']:
            if getattr(args, arg_name) is not None:
                root_logger.warning(f"Ignoring {arg_name}. Precomputed FDRs are provided")
    elif args.precomp_cutcounts is not None:
        if args.bam is not None:
            root_logger.warning("Ignoring bam file. Precomputed cutcounts are provided")
    elif args.bam is None:
        parser.error("Either --bam or --precomp_cutcounts or --precomp_fdrs should be provided")

    return args, logger_level


if __name__ == "__main__":
    main()
