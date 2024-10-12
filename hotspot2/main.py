import logging
import gc
import argparse
from processors import GenomeProcessor, root_logger, set_logger_config
from utils import read_chrom_sizes, df_to_tabix


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
        chromosomes=['chr20', 'chr19']
    )
    precomp_fdrs = args.precomp_fdrs

    if precomp_fdrs is None:
        root_logger.info('Calculating p-values')
        pvals_data = genome_processor.calc_pval(args.cutcounts)
        root_logger.debug('Saving P-values')
        parquet_path = f"{args.prefix}.stats.parquet"
        pvals_data.data_df.to_parquet(
            parquet_path,
            engine='pyarrow',
            compression='zstd',
            compression_level=22,
            index=False,
            partition_cols=['chrom'],
        )
        pvals_data.params_df.to_csv(f"{args.prefix}.params.gz", sep='\t', index=False)
        del pvals_data
        gc.collect()

        precomp_fdrs = parquet_path

    root_logger.info('Calling hotspots')
    hotspots = genome_processor.call_hotspots(precomp_fdrs, fdr_tr=args.fdr).data_df
    hotspots_path = f"{args.prefix}.hotspots.fdr{args.fdr}.bed.gz"
    df_to_tabix(hotspots, hotspots_path)
    del hotspots
    gc.collect()

    root_logger.info('Calling peaks')
    peaks = genome_processor.call_peaks(hotspots_path, args.cutcounts).data_df
    peaks_path = f"{args.prefix}.peaks.fdr{args.fdr}.bed.gz"
    df_to_tabix(peaks, peaks_path)

    if args.save_density:
        root_logger.info('Computing densities')
        density_data = genome_processor.calc_density(args.cutcounts).data_df
        root_logger.debug('Saving densities')
        denisty_path = f"{args.prefix}.density.bed.gz"
        df_to_tabix(density_data, denisty_path)
    root_logger.info('Program finished')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Call hotspots from cutcounts")
    
    # common arguments
    parser.add_argument("prefix", type=str, help="Output prefix")
    parser.add_argument("cutcounts", help="Path to cutcounts tabix file")

    parser.add_argument("--chrom_sizes", help="Path to chromosome sizes file. If none assumed to be hg38 sizes", default=None)
    parser.add_argument("--fdr", help="FDR threshold for p-values", type=float, default=0.05)
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use", default=1)
    parser.add_argument("--debug", help="Path to chromosome sizes file. If none assumed to be hg38 sizes", action='store_true', default=False)

    # Arguments for calculating p-values
    parser.add_argument("--mappable_bases", help="Path to mappable bases file (if needed)", default=None)
    parser.add_argument("--window", help="Window size for smoothing cutcounts", type=int, default=201)
    parser.add_argument("--background_window", help="Background window size", type=int, default=50001)
    
    # Arguments to call hotspots, skip calculating p-values if provided
    parser.add_argument("--precomp_fdrs", help="Path to pre-calculated partitioned parquet file(s) with FDRs. Skips FDR calculation", default=None)

    # Optional - save density
    parser.add_argument("--save_density", action='store_true', help="Save density of cutcounts", default=False)

    args = parser.parse_args()
    logger_level = logging.DEBUG if args.debug else logging.INFO
    set_logger_config(root_logger, logger_level)

    if args.precomp_fdrs is not None and args.mappable_bases is not None:
        root_logger.warning("Ignoring mappable bases file as precalculated FDRs are provided")

    return args, logger_level


if __name__ == "__main__":
    main()
