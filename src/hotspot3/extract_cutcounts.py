import argparse
import logging
import sys


epilog = """
Writes {id}.cutcounts.bed.gz (tabix-indexed per-base cut counts) and {id}.total_cutcounts to --outdir.
Both can be passed to hotspot3 with --cutcounts to skip this step.

Example:
hotspot3-cutcounts sample123 --bam sample123.bam --chrom_sizes hg38.chrom.sizes --outdir sample123

See full documentation at: https://github.com/vierstralab/hotspot3
"""

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Extract per-base cut counts (5' ends of reads) from a BAM/CRAM file.",
        epilog=epilog
    )
    parser.add_argument("id", type=str, help="Unique identifier of the sample")
    parser.add_argument("--bam", help="Path to input bam/cram file", required=True)
    parser.add_argument(
        "--chrom_sizes",
        help="(Required) Path or URL to chromosome sizes file",
        required=True
    )
    parser.add_argument(
        "--reference",
        help="Path to reference fasta file. Required to work with cram files with missing fasta", default=None
    )
    parser.add_argument(
        "--outdir",
        help="Path to output directory",
        default="."
    )
    parser.add_argument(
        "--cpus",
        type=int,
        help="Number of CPUs to use. Doesn't use more than # of chromosomes.",
        default=1
    )
    parser.add_argument(
        "--chromosomes",
        help="List of chromosomes to process. Useful for debug",
        nargs='+',
        default=None
    )
    parser.add_argument(
        "--debug",
        help="Adds additional prints",
        action='store_true',
        default=False
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Imports here to make help message appear faster
    from hotspot3.io.logging import setup_logger
    logger_level = logging.DEBUG if args.debug else logging.INFO
    root_logger = setup_logger(level=logger_level)
    root_logger.info(f"Executing command: {' '.join(sys.argv)}")

    from hotspot3.config import ProcessorConfig
    from hotspot3.processors import GenomeProcessor
    from hotspot3.io.paths import Hotspot3Paths

    config = ProcessorConfig(cpus=args.cpus, logger_level=logger_level)
    genome_processor = GenomeProcessor(
        sample_id=args.id,
        chrom_sizes_file=args.chrom_sizes,
        chromosomes=args.chromosomes,
        reference_fasta=args.reference,
        config=config,
    )
    paths = Hotspot3Paths(outdir=args.outdir, sample_id=args.id)
    genome_processor.extract_cutcounts_from_bam(args.bam, paths.cutcounts)
    genome_processor.get_total_cutcounts(paths.cutcounts, paths.total_cutcounts)

    root_logger.info('Program finished')


if __name__ == "__main__":
    main()
