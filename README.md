# Calling peaks with hotspot3

- Peak calling assumes a negative binomial distribution for background cut counts.
- A segmentation-based approach (see [BABACHI](https://github.com/autosome-ru/BABACHI)) is used to identify regions with approximately uniform background signal. Within each segment, background parameters are estimated using the negative binomial model.


# Command line interface
- `hotspot3` - Call peaks from input data using MODWT-smoothed signal and local background estimation.
- `hotspot3-pvals` - (Experimental) Extract raw p-values for regions defined in a reference BED file. This is useful when comparing or aggregating signal across multiple samples at predefined loci (e.g., shared peak positions), enabling downstream consensus ("core") peak calling.
# Installation
## Prerequisites
`conda` (recommended: `mamba` for faster solves)
`python ≥ 3.8`
## 1. Create conda environment from environment.yml
### For linux:
```mamba env create -n hotspot3 -f environment.yml
conda activate hotspot3
```
### For macOS:
```mamba env create -n hotspot3 -f environment.macos.yml
conda activate hotspot3
```
## 2. Install hotspot3
Once the environment is activated:
```
git clone https://github.com/autosome-ru/hotspot3.git
cd hotspot3
pip install -e .
```
Not on PyPI yet — coming soon!

## 3. Test installation
You can check that the CLI is working with:
```
hotspot3 --help
```
# Example Usage
## Basic peak calling from BAM/CRAM
```
hotspot3 AG10883 \
  --chrom_sizes /net/seq/data2/projects/sabramov/hotspot3/GRCh38_no_alts.nuclear.chrom_sizes \
  --bam AG10883.filtered.cram \
  --fdrs 0.01 0.05 0.1 \
  --outdir ./AG10883 \
  --save_density \
  --cpus 6
```
This performs processing from CRAM, calls peaks at 1%, 5%, and 10% FDR, and saves cutcount density. Uses 6 threads for parallel processing.

## Additional FDR threshold (fast reuse of precomputed data)
```
hotspot3 AG10883 \
  --chrom_sizes /net/seq/data2/projects/sabramov/hotspot3/GRCh38_no_alts.nuclear.chrom_sizes \
  --cutcounts ./AG10883/AG10883.cutcounts.bed.gz \
  --signal_parquet ./AG10883/debug/AG10883.smoothed_signal.parquet \
  --fdrs_parquet ./AG10883/debug/AG10883.fdrs.parquet \
  --fdrs 0.001 \
  --outdir ./AG10883 \
  --cpus 6
```
This reuses previously computed intermediate files to quickly generate peaks at an additional FDR threshold (e.g., 0.001). Runtime is ~30 seconds.

# Input parameters
## Required arguments
- `sample_id` - Unique identifier for the sample (used for naming outputs).
- `--chrom_sizes CHROM_SIZES` - Two-column TSV file with chromosome names and sizes (no header). Local path or URL.
- `--bam BAM` - Path to the input BAM or CRAM file.

- `--fdrs FDRS [FDRS ...]` - Space separated list of FDR thresholds to generate peak calls at. A Parquet track of per-base FDR values is generated only for the largest FDR threshold among provided.

## Arguments to skip steps using pre-calculated data

- `--cutcounts CUTCOUNTS` - Tabix-indexed file with per-base cut counts. Skips extracting cut counts from BAM/CRAM.
- `--signal_parquet SIGNAL_PARQUET` - Path to pre-calculated partitioned parquet file(s) with per-bp smoothed signal. Skips MODWT signal smoothing
- `--pvals_parquet PVALS_PARQUET` - Path to pre-calculated partitioned parquet file(s) with per-bp p-values. Skips p-value calculation
- `--fdrs_parquet FDRS_PARQUET` - Path to pre-calculated fdrs. (Experimental) Can be used with multiple_samples_fdr.py to correct across samples.

## Optional arguments
- `--cpus CPUS` - Number of CPUs to use. High thread count increases memory usage. No benefit from using more threads than chromosomes.
  
- `--reference REFERENCE` - Path to reference FASTA (required for CRAMs missing sequence dictionary).
- `--mappable_bases MAPPABLE_BASES` - Tabix-indexed BED file listing mappable positions.

- `--chromosomes CHROMOSOMES [CHROMOSOMES ...]` - Restrict to specific chromosomes (for debugging).
- `--save_density` -  Save normalized cut count density as output.

- `--debug` - Enable debug mode (extra logging).
- `--outdir OUTDIR` - Path to output directory
- `--tempdir TEMPDIR` - Path to temp directory. Defaults to the system temp location.

## Change if you know what you are doing
- `--window WINDOW` - Smoothing window size for cut counts.
- `--background_window BACKGROUND_WINDOW` - Background window size
- `--signal_quantile SIGNAL_QUANTILE` - Fraction of genomic positions to model as a background (default: 0.995). Used to exclude extreme outliers (e.g., regions with unusually high coverage) when fitting the background model. This reduces computational burden by limiting the background estimation to the lower portion of the signal distribution (e.g., the lowest 99.5% of sites if `--signal_quantile 0.995`).


# Output files
Currently, Hotspot3 doesn't delete files in the debug folder upon completion. You can manually delete the created `debug` folder to save disk space.

- tabix indexed cutcounts: `{sample_id}.cutcounts.bed.gz` (~200MB)
- File with total # of cutcounts: `{sample_id}.total_cutcounts` (~10kb)

- tabix indexed per-segment fit stats: `{sample_id}.fit_stats.tsv.gz` (~20MB)

- per-bp raw p-values: `{sample_id}.pvals.parquet` (large, ~1.5GB)

For each FDR threshold:
  - tabix indexed hotspots at FDR: `{sample_id}.hotspots.fdr{fdr}.bed.gz`
  - hotspots at FDR in bb (BED12) format
  - tabix indexed peaks at FDR: `{sample_id}.peaks.fdr{fdr}.bed.gz`
  - peaks at FDR in bb (BED12) format


The following files are saved to the debug folder:

    - per-bp smoothed signal: `{sample_id}.smoothed_signal.parquet` (large, ~2GB)
    - estimated parameters background fits: `{sample_id}.fit_params.parquet` (large, ~2GB)
    - per-bp FDR estimates: `{sample_id}.fdrs.parquet` (~600MB)

# Performance and Resource Requirements
- Typical runtime: ~1 hour for a 100 million read CRAM file.

- CPU usage: Parallelized by chromosome. Using more CPUs (e.g., 22 for all chromosomes) can reduce wall-clock time but significantly increases memory usage.

- Memory usage: ~80 GB RAM with 6 CPUs. Can exceed 150 GB when using 22 threads (e.g., one per chromosome).

- Recommendation: Use a machine with 100–200 GB RAM for high-threaded runs (significantly improves time). Reduce thread count to lower memory use.

# Authors

Developed by:

- Sergey Abramov
- Alexandr Boytsov
- Jeff Vierstra

Altius Institute for Biomedical Sciences
