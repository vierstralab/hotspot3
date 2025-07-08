import argparse
import logging
import sys


epilog = """
Examples:
hotspot3 sample123 --bam sample123.bam --chrom_sizes hg38.chrom.sizes --fdrs 0.01 0.05
hotspot3 sample123 --chrom_sizes hg38.chrom.sizes --cutcounts sample123.cutcounts.bed.gz --signal_parquet sample123.signal.parquet --fdrs_parquet sample123.fdrs.parquet --fdrs 0.001

See full documentation at: https://github.com/vierstralab/hotspot3
"""

def parse_arguments(extra_desc: str = "") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Call peaks using hotspot3 with local background estimation and MODWT smoothing.",
        epilog=epilog + extra_desc
    )
    
    # common arguments
    parser.add_argument("id", type=str, help="Unique identifier of the sample")
    parser.add_argument("--bam", help="Path to input bam/cram file", default=None)

    parser.add_argument(
        "--chrom_sizes", 
        help="Path to chromosome sizes file. If none assumed to be hg38 sizes",
        default=None
    )
    parser.add_argument(
        "--fdrs", 
        help="List of FDR thresholds, space separated",
        type=float, 
        nargs='+',
        default=[0.05]
    )
    parser.add_argument(
        "--reference",
        help="Path to reference fasta file. Required to work with cram files with missing fasta", default=None
    )
    parser.add_argument(
        "--cpus",
        type=int,
        help="Number of CPUs to use. Doesn't use more than # of chromosomes.",
        default=1
    )
    parser.add_argument(
        "--outdir",
        help="Path to output directory",
        default="."
    )
    parser.add_argument(
        "--debug",
        help="Adds additional prints",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--tempdir",
        help="Path to temporary directory. Defaults to system temp directory",
        default=None
    )

    parser.add_argument(
        "--mappable_bases", 
        help=" (Optional) Path to tabix indexed mappable bases file. Ignore unmappable bases for statistical model.",
        default=None
    )
    parser.add_argument(
        "--window",
        help="Window size for smoothing cutcounts",
        type=int,
        default=151
    )
    parser.add_argument(
        "--background_window",
        help="Background window size",
        type=int,
        default=50001
    )
    parser.add_argument(
        "--signal_quantile",
        help="Max proportion of background expected in the data. Change if you know what you are doing.",
        type=float,
        default=0.995
    )
    
    # Arguments to skip previous steps if provided

    parser.add_argument(
        "--cutcounts",
        help="Path to pre-calculated cutcounts tabix file. Skip extracting cutcounts from bam file",
        default=None
    )
    parser.add_argument(
        "--signal_parquet",
        help="Path to pre-calculated partitioned parquet file(s) with per-bp smoothed signal. Skips modwt signal smoothing",
        default=None
    )

    parser.add_argument(
        "--pvals_parquet",
        help="Path to pre-calculated partitioned parquet file(s) with per-bp p-values. Skips p-value calculation",
        default=None
    )

    parser.add_argument(
        "--fdrs_parquet",
        help="Path to pre-calculated fdrs. Can correct for several samples using multiple_samples_fdr.py",
        default=None
    )

    parser.add_argument(
        "--chromosomes",
        help="List of chromosomes to process. Useful for debug",
        nargs='+',
        default=None
    )

    # Optional - save density
    parser.add_argument(
        "--save_density",
        action='store_true',
        help="Save normalized density of cutcounts",
        default=False
    )

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_arguments()

    # Imports here to make help message faster
    from hotspot3.io.logging import setup_logger
    logger_level = logging.DEBUG if args.debug else logging.INFO
    root_logger = setup_logger(level=logger_level)
    root_logger.info(f"Executing command: {' '.join(sys.argv)}")


    from hotspot3.config import ProcessorConfig
    from hotspot3.from_configs import GenomeProcessor, run_from_configs
    from hotspot3.io.paths import Hotspot3Paths

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

    paths = Hotspot3Paths(
        outdir=args.outdir,
        sample_id=args.id,
        save_density=args.save_density,
        bam=args.bam,
        cutcounts=args.cutcounts,
        smoothed_signal=args.signal_parquet,
        pvals=args.pvals_parquet,
        fdrs=args.fdrs_parquet,
    )
    run_from_configs(genome_processor, paths, args.fdrs)
    
    root_logger.info('Program finished')


if __name__ == "__main__":
    main()
