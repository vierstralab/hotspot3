import logging
import gc
import argparse
from processors import GenomeProcessor, root_logger, set_logger_config
from utils import read_chrom_sizes


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
    fdr_path = args.precalc_fdrs
    if fdr_path is None:
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

        fdr_path = parquet_path
    
    precalc_density = args.precalc_density
    if precalc_density is None:
        root_logger.info('Calling peaks')
        density_data = genome_processor.calc_density(args.cutcounts)
        precalc_density = f"{args.prefix}.density.bed"
        density_data.data_df.to_csv(precalc_density, sep='\t', index=False)
        del density_data
        gc.collect()

    root_logger.info('Calling hotspots')
    hotspots_path = f"{args.prefix}.hotspots.bed"
    hotspots = genome_processor.call_hotspots(fdr_path, fdr_tr=args.fdr)
    hotspots.data_df.to_csv(hotspots_path, sep='\t', index=False) # TODO save as tabix

    root_logger.info('Calling peaks is not yet implemented')
    #genome_processor.call_peaks(hotspots_path, precalc_density)
    root_logger.info('Program finished')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Call hotspots from cutcounts")
    
    # common arguments
    parser.add_argument("prefix", type=str, help="Output prefix")
    parser.add_argument("--chrom_sizes", help="Path to chromosome sizes file. If none assumed to be hg38 sizes", default=None)
    parser.add_argument("--fdr", help="FDR threshold for p-values", type=float, default=0.05)
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use", default=1)
    parser.add_argument("--debug", help="Path to chromosome sizes file. If none assumed to be hg38 sizes", action='store_true', default=False)

    # Arguments for calculating p-values
    parser.add_argument("--cutcounts", help="Path to cutcounts tabix file")
    parser.add_argument("--mappable_bases", help="Path to mappable bases file (if needed)", default=None)
    parser.add_argument("--window", help="Window size for smoothing cutcounts", type=int, default=201)
    parser.add_argument("--background_window", help="Background window size", type=int, default=50001)
    
    # Arguments to call hotspots, skip calculating p-values if provided
    parser.add_argument("--precalc_fdrs", help="Path to pre-calculated partitioned parquet file(s) with FDRs. Skips FDR calculation", default=None)

    # Arguments to call peaks, skip calculating p-values if provided
    parser.add_argument("--precalc_density", help="Path to pre-calculated tabix density file", default=None)

    args = parser.parse_args()
    logger_level = logging.DEBUG if args.debug else logging.INFO
    set_logger_config(root_logger, logger_level)

    if args.precalc_fdrs is not None:
        if args.cutcounts is not None:
            root_logger.warning("Ignoring cutcounts file as precalculated FDRs are provided")
        if args.mappable_bases is not None:
            root_logger.warning("Ignoring mappable bases file as precalculated FDRs are provided")
    
    if args.precalc_density is not None:
        if args.cutcounts is not None:
            root_logger.warning("Ignoring cutcounts file as precalculated density is provided")
    # elif args.cutcounts is None:
    #     raise ValueError("Either cutcounts or precalculated density file should be provided for calling peaks")

    return args, logger_level


if __name__ == "__main__":
    main()
