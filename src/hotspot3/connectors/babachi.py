from hotspot3.models import GlobalFitResults
from hotspot3.logging import WithLogger
from hotspot3.connectors.bottleneck import BottleneckWrapper
from babachi.segmentation import GenomeSegmentator
from babachi.models import GenomeSNPsHandler, ChromosomeSNPsHandler
import numpy as np
import numpy.ma as ma
from typing import List
from genome_tools.genomic_interval import GenomicInterval


class BabachiWrapper(WithLogger):

    def run_segmentation(self, agg_cutcounts: ma.MaskedArray, per_window_trs: np.ndarray, global_fit: GlobalFitResults, chrom_name, chrom_size):
        step = self.config.babachi_segmentation_step

        bn_wrapper = BottleneckWrapper(config=self.config, logger=self.logger)
        assumed_signal_mask = bn_wrapper.filter_by_tr_spatially(agg_cutcounts, per_window_trs)
        background = agg_cutcounts.filled(np.nan)[::step]
        background[assumed_signal_mask[::step]] = np.nan
        starts = np.arange(0, len(background), dtype=np.uint32) * step

        chrom_bad = (1 - global_fit.p) / global_fit.p
        mult = np.linspace(1, 10, 20)
        bads = [*(mult * chrom_bad), *(1 / mult[1:] * chrom_bad)]

        valid_counts = ~np.isnan(background)

        chrom_handler = ChromosomeSNPsHandler(
            chrom_name,
            positions=starts[valid_counts], 
            read_counts=np.stack(
                [
                    np.full(background.shape[0], global_fit.r, dtype=np.float32),
                    background
                ]
            ).T[valid_counts, :]
        )
        snps_collection = GenomeSNPsHandler(chrom_handler)

        gs = GenomeSegmentator(
            snps_collection=snps_collection,
            chrom_sizes={chrom_name: chrom_size},
            jobs=1,
            logger_level=self.config.logger_level,
            segmentation_mode='binomial',
            states=bads,
            logger=self.logger,
            allele_reads_tr=0,
            b_penalty=9,
            min_seg_bp=5000,
            min_seg_snps=0,
            subchr_filter=0
        )
        bad_segments = gs.estimate_BAD()
        return [GenomicInterval(x.chr, x.start, x.end, BAD=x.BAD / chrom_bad) for x in bad_segments]
    
    def annotate_with_segments(self, shape, bad_segments: List[GenomicInterval]):
        babachi_result = np.zeros(shape, dtype=np.float16)
        for segment in bad_segments:
            babachi_result[segment.start:segment.end] = segment.BAD
        return babachi_result
