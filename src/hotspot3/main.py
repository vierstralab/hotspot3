import logging
import argparse
import os
import sys
import shutil

from hotspot3.processors import GenomeProcessor
from hotspot3.io.logging import setup_logger
from hotspot3.config import ProcessorConfig
from hotspot3.io.paths import Hotspot3Paths
import networkx as nx


def resolve_required_steps(outputs, available, graph):
    """
    Given a set of desired outputs and already available nodes,
    return the minimal set of required nodes (topologically sorted).
    """
    required = set()
    visited = set()

    def visit(node):
        if node in visited or node in available:
            return
        visited.add(node)
        for dep in graph.predecessors(node):
            visit(dep)
        required.add(node)

    for out in outputs:
        visit(out)

    return list(nx.topological_sort(graph.subgraph(required)))


def find_missing_steps(paths: Hotspot3Paths, save_density):
    step_graph = nx.DiGraph()
    step_graph.add_edges_from([
        ("bam", "cutcounts"),
        ("cutcounts", "total_cutcounts"),
        ("cutcounts", "smoothed_signal"),
        ("total_cutcounts", "smoothed_signal"),
        ("cutcounts", "fit_params"),
        ("total_cutcounts", "fit_params"),
        ("fit_params", "pvals"),
        ("cutcounts", "pvals"),
        ("pvals", "fdrs"),
        ("smoothed_signal", "normalized_density"),
        ("fdrs", "peak_calling"),
        ("smoothed_signal", "peak_calling")
    ])
    available = {x for x in step_graph.nodes if paths.was_set(x)}
    outputs = {'peak_calling',}
    if save_density:
        outputs.add('normalized_density')
    result = resolve_required_steps(outputs, available, step_graph)
    if 'bam' in result:
        raise ValueError("Provide a bam file or cutcounts")
    return result


def run_from_configs(
        genome_processor: GenomeProcessor, 
        paths: Hotspot3Paths, 
        fdrs, 
        save_density,
    ):
    step_names = find_missing_steps(paths, save_density)
    genome_processor.logger.info(f"Running: {', '.join(step_names)}")
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

    paths = Hotspot3Paths(
        outdir=args.outdir,
        sample_id=args.id,
        cutcounts=args.cutcounts,
        smoothed_signal=args.signal_parquet,
        pvals=args.pvals_parquet,
        fdrs=args.fdrs_parquet,
        
    )
    root_logger.info(f"Executing command: {' '.join(sys.argv)}")
    run_from_configs(genome_processor, paths, args.fdrs, args.save_density)
    
    root_logger.info('Program finished')


def parse_arguments(extra_desc: str = "") -> argparse.Namespace:
    name = "Run hotspot3 peak calling" + extra_desc
    # TODO: Change description to reflect the new functionality
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=name + """
    
    Saves the following files in the output directory:
        - tabix indexed cutcounts: {sample_id}.cutcounts.bed.gz (~200 MB)
        - file with total # of cutcounts: {sample_id}.total_cutcounts (~1 kb)
        - per-bp 0.995 percentile of 'background': {sample_id}.background.bw (~10 MB)
        - per-bp raw p-values: {sample_id}.pvals.parquet (large, ~800 MB)
        Optional if --save_density is provided:
            - normalized density of cutcounts in bigwig format: {sample_id}.normalized_density.bw
    
    Additionally, in debug folder saves:
        - per-bp smoothed signal: {sample_id}.smoothed_signal.parquet (large, ~10GB)
        - parameters used for background per-chromosome fits: {sample_id}.fit_params.parquet (large, ~10GB)
        - per-bp threshold for background fit: {sample_id}.thresholds.bw (~10 MB)
        - per-bp 0.995 background estimated for each segment: {sample_id}.per_segment.background.bw (~10 MB)
        - per-bp FDR estimates: {sample_id}.fdrs.parquet (~200 MB)
    
    Note:
        Multiple FDR thresholds can be provided as a space-separated list.

    For each FDR threshold:
        - tabix indexed hotspots at FDR: {sample_id}.hotspots.fdr{fdr}.bed.gz
        - bb hotspots at FDR: {sample_id}.hotspots.fdr{fdr}.bb

        - tabix indexed peaks at FDR: {sample_id}.peaks.fdr{fdr}.bed.gz
        - bb peaks at FDR: {sample_id}.peaks.fdr{fdr}.bb
    
    To quickly re-run hotspot and peak calling at other FDRs without recomputing everything:
    Fastest option (everything precomputed):
    Provide both
        --fdrs_parquet {sample_id}.fdrs.parquet
        and
        --signal_parquet {sample_id}.smoothed_signal.parquet
    if you still have the smoothed signal. 

    Slower fallback (signal needs recomputing):
    If signal_parquet is missing, you can still provide
        --fdrs_parquet {sample_id}.fdrs.parquet
        and either
        --cutcounts {sample_id}.cutcounts.bed.gz
        or
        --bam input.bam
    to regenerate the smoothed signal before peak calling.
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
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use. Doesn't use more than # of chromosomes.", default=1)
    parser.add_argument("--outdir", help="Path to output directory", default=".")
    parser.add_argument("--debug", help="Adds additional prints", action='store_true', default=False)
    parser.add_argument("--tempdir", help="Path to temporary directory. Defaults to system temp directory", default=None)

    # Arguments for calculating p-values
    parser.add_argument("--mappable_bases", help=" (Optional) Path to tabix indexed mappable bases file. Ignore unmappable bases for statistical model.", default=None)
    parser.add_argument("--window", help="Window size for smoothing cutcounts", type=int, default=151)
    parser.add_argument("--background_window", help="Background window size", type=int, default=50001)
    parser.add_argument(
        "--signal_quantile",
        help="Max proportion of background expected in the data. Change if you know what you are doing.",
        type=float,
        default=0.995
    )
    
    # Arguments to skip previous steps if provided
    parser.add_argument("--bam", help="Path to input bam/cram file", default=None)
    parser.add_argument("--cutcounts", help="Path to pre-calculated cutcounts tabix file. Skip extracting cutcounts from bam file", default=None)
    parser.add_argument("--signal_parquet", help="Path to pre-calculated partitioned parquet file(s) with per-bp smoothed signal. Skips modwt signal smoothing", default=None)
    parser.add_argument("--pvals_parquet", help="Path to pre-calculated partitioned parquet file(s) with per-bp p-values. Skips p-value calculation", default=None)
    parser.add_argument("--fdrs_parquet", help="Path to pre-calculated fdrs. Can correct for several samples using multiple_samples_fdr.py", default=None)

    parser.add_argument("--chromosomes", help="List of chromosomes to process. Useful for debug", nargs='+', default=None)

    # Optional - save density
    parser.add_argument("--save_density", action='store_true', help="Save normalized density of cutcounts", default=False)

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
