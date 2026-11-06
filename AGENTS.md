# hotspot3 — notes for coding agents

Peak caller for DNase-seq / ATAC-seq without a control. Per sample: BAM/CRAM -> per-base cut
counts -> MODWT-smoothed signal -> negative binomial background fitted per genomic segment
(segments from BABACHI) -> per-base p-values -> FDR -> hotspots and peaks at each FDR.
User docs are in `README.md` (CLI arguments, output files, how to read `fit_stats`).

Everything is parallel by chromosome. Downstream: visualization with genome-tools (https://github.com/vierstralab/genome-tools/blob/master/AGENTS.md) plot components
(`genome_tools.plotting.modular_plot.plot_components.hotspot3`).

## Where things are

| Need | Look in |
|---|---|
| CLI arguments -> objects | `src/hotspot3/cli.py` (`hotspot3` entry point) |
| Order of pipeline steps | `src/hotspot3/from_configs.py` (`run_from_configs`) |
| Which steps run, given the inputs provided | `src/hotspot3/io/paths.py` (`Hotspot3Paths`, `StepsSolver`) |
| All tunable parameters and defaults | `src/hotspot3/config.py` (`ProcessorConfig`) |
| Genome-wide steps, process pool | `src/hotspot3/processors.py` (`GenomeProcessor`) |
| Per-chromosome work | `src/hotspot3/processors.py` (`ChromosomeProcessor`) |
| BAM/CRAM -> cut counts | `src/hotspot3/connectors/bam2bed.py`, `src/hotspot3/scripts/extract_cutcounts.sh` |
| Segmentation | `src/hotspot3/connectors/babachi.py` |
| Background fit | `src/hotspot3/background_fit/` (`defunc.py` is unused) |
| p-values, FDR | `src/hotspot3/scoring/` |
| Smoothing | `src/hotspot3/signal_smoothing/modwt.py` |
| Hotspots and peaks | `src/hotspot3/peak_calling/peak_calling.py` |
| Reading / writing parquet, tabix, bigWig, bigBed | `src/hotspot3/io/readers.py`, `src/hotspot3/io/writers.py` |
| Other CLIs | `extract_cutcounts.py` (`hotspot3-cutcounts`), `extract_pvals.py` (`hotspot3-pvals`, experimental), `ucsc_trackhub.py` (`hotspot3-trackhub`), `track_memory.py` (`hotspot3-track-mem`, needs the `dev` extra: psutil) |
| Unfinished, not installed | `multiple_sample_fdr.py` (entry point commented out in `pyproject.toml`) |

## Pipeline and step skipping

Steps, in `run_from_configs` order: `cutcounts` -> `total_cutcounts` -> `smoothed_signal` ->
`fit_params` (+ thresholds and background bigWigs) -> `pvals` -> `fdrs` ->
`normalized_density` (only with `save_density`) -> `peak_calling`.

- `StepsSolver` builds the step graph and runs only the steps not covered by inputs passed
  explicitly (`--cutcounts`, `--signal_parquet`, `--pvals_parquet`, `--fdrs_parquet`).
- Files already in `outdir` are **not** detected: re-running without these arguments recomputes
  everything.
- `--cutcounts` implies `{sample_id}.total_cutcounts` next to it.
- The FDR parquet holds values only up to `max(fdrs)`; higher thresholds later need `--pvals_parquet`.
- Peak calling deletes and recreates `{outdir}/fdr{fdr}/` for every FDR requested.

## Running from Python

The CLI only builds `ProcessorConfig`, `GenomeProcessor` and `Hotspot3Paths` and calls
`run_from_configs`; the README section "Running from Python" has the example.

- `GenomeProcessor(chromosomes=['chr21', 'chr22'])` restricts the run, e.g. for a quick check.
- `Hotspot3Paths` keyword arguments mirror the skip arguments: `bam`, `cutcounts`,
  `smoothed_signal`, `pvals`, `fdrs`. Unknown keys raise; provided paths must exist; `outdir` is
  created if missing.
- The `if __name__ == "__main__":` guard is needed in scripts because workers are spawned
  (macOS; also the default on Linux from Python 3.14).
- `cpus > 1` uses `ProcessPoolExecutor`; `GenomeProcessor` and `ChromosomeProcessor` are pickled
  to the workers, so keep them picklable (no open file handles or lambdas as attributes).
- In Jupyter the `__main__` guard is not needed.
- Logs go to stdout through the `hotspot3` logger (`setup_logger` in `io/logging.py`);
  `ProcessorConfig(logger_level=logging.DEBUG)` for more.

## External programs

Called through `subprocess`, so they must be on PATH (all in `environment.yml`):
- `samtools`, `bam2bed` and `sort-bed` (bedops), `bgzip`, `mawk` or `awk`: cut-count extraction
  (`scripts/extract_cutcounts.sh`);
- `bedToBigBed`: hotspot and peak bigBeds.

BigWigs are written with pyBigWig, tabix files with `genome_tools.helpers.df_to_tabix`.

## Outputs

Names and sizes are in `README.md` ("Output files"). For code that reads them:
- Parquet outputs (`pvals`, `fdrs`, `smoothed_signal`, `fit_params`) are datasets partitioned by
  `chrom`, one row per base. Read one chromosome with `ChromReader(genomic_interval=...).extract_from_parquet(path, columns=[...])`;
  it raises `NotEnoughDataForContig` if the chromosome is missing.
- Peaks and hotspots: tabix BED plus bigBed (BED12) per FDR, in `fdr{fdr}/`; columns are listed
  in the README.

## Gotchas

- `--chrom_sizes` is required: `chrom_sizes=None` raises `NotImplementedError` (no embedded default).
- Mappable bases: a chromosome missing from the file is skipped (`NotEnoughDataForContig`), while
  coordinates beyond `chrom_sizes` raise `ValueError`.
- `--debug` both raises the log level and sets `save_debug`.
- Memory scales with `cpus` (~80 GB at 6, >150 GB at 24): one chromosome per worker is held in
  memory at full per-base resolution.
- `numpy<2.0` and `babachi==3.0.0` are pinned in `pyproject.toml`; don't bump them as a side
  effect of other changes.

## Working in this repo

- Match the surrounding style: classes inherit `WithLogger` / `WithLoggerAndInterval` and are
  created with `copy_with_params` so config and logger propagate; per-chromosome methods return
  `ProcessorOutputData` and are wrapped with `parallel_func_error_handler`.
- New CLI options go in `cli.py`, the `ProcessorConfig` field (if a model parameter) and the
  README "Input parameters" section.
- There is no test suite in git. `data/` and `test/` are git-ignored local folders; a quick end-to-end
  run on chr21/chr22 (`--chromosomes chr21 chr22`) is the practical check.
