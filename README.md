# Calling peaks with hotspot3

- Peak calling assuming negative binomial distribution of cutcounts in the background
- Uses segmentation approach (see BABACHI) to find regions with approximately uniform background and estimates parameters of background negative binomial distribution

## Command line interface
- hotspot3 - call peaks
- hotspot3-pvals - extract pvals using reference bed file

## Input parameters
- sample_id - unique identifier of the sample


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