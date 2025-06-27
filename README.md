# Calling peaks with hotspot3

- Peak calling assuming negative binomial distribution of cutcounts in the background
- Uses segmentation approach (see BABACHI) to find regions with approximately uniform background and estimates parameters of background negative binomial distribution

## Command line interface
- `hotspot3` - call peaks
- `hotspot3-pvals` - (not fully tested) extract pvals using reference bed file

## Input parameters
### Required arguments
- `sample_id` - Unique identifier of the sample
- `--chrom_sizes CHROM_SIZES` - Two column tsv file (chrom size), no header. Can provide url.
- `--bam BAM` - Path to input bam/cram file

- `--fdrs FDRS [FDRS ...]` - List of FDR thresholds, space separated

### Arguments to skip steps using pre-calculated data

- `--cutcounts CUTCOUNTS` - Path to pre-calculated cutcounts tabix file. Skip extracting cutcounts from bam file
- `--signal_parquet SIGNAL_PARQUET` - Path to pre-calculated partitioned parquet file(s) with per-bp smoothed signal. Skips modwt signal smoothing
- `--pvals_parquet PVALS_PARQUET` - Path to pre-calculated partitioned parquet file(s) with per-bp p-values. Skips p-value calculation
- `--fdrs_parquet FDRS_PARQUET` - Path to pre-calculated fdrs. Can correct for several samples using multiple_samples_fdr.py

### Optional arguments
- `--reference REFERENCE` - Path to fasta file (for cram files with path to fasta missing)
- `--mappable_bases MAPPABLE_BASES` - Tabix indexed bed file listing mappable bases

- `--chromosomes CHROMOSOMES [CHROMOSOMES ...]` - Space separated list of chromosomes to process. Useful for debug
- `--save_density` - Save normalized density of cutcounts

- `--cpus CPUS` - # of cpus to use. Uses a lot of memory for large # of cpus. Doesn't utlize more than number of chromosomes
- `--debug` - Add additional prints and save tmp files
- `--outdir OUTDIR` - Path to output directory
- `--tempdir TEMPDIR` - Path to temporary directory. Defaults to system temp directory

### Change if you know what you are doing
- `--window WINDOW` - Window size for smoothing cutcounts
- `--background_window BACKGROUND_WINDOW` - Background window size
- `--signal_quantile SIGNAL_QUANTILE` - Max proportion of background expected in the data
  


## Output files
- tabix indexed cutcounts: {sample_id}.cutcounts.bed.gz (~200MB)
- File with total # of cutcounts: {sample_id}.total_cutcounts (~10kb)

- tabix indexed per-segment fit stats: {sample_id}.fit_stats.tsv.gz (~20MB)

- per-bp raw p-values: {sample_id}.pvals.parquet (large, ~1.5GB)

For each FDR threshold:

    - tabix indexed hotspots at FDR: {sample_id}.hotspots.fdr{fdr}.bed.gz
    - hotspots at FDR in bb (BED12) format
    - tabix indexed peaks at FDR: {sample_id}.peaks.fdr{fdr}.bed.gz
    - peaks at FDR in bb (BED12) format


If --debug is enabled, the following files are saved:

    - per-bp smoothed signal: {sample_id}.smoothed_signal.parquet (large, ~2GB)
    - estimated parameters background fits: {sample_id}.fit_params.parquet (large, ~2GB)
    - per-bp FDR estimates: {sample_id}.fdrs.parquet (~600MB)