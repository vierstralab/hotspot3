import numpy as np
import numpy.ma as ma
from typing import List

from genome_tools import GenomicInterval

from hotspot3.helpers.models import FitResults, NotEnoughDataForContig
from hotspot3.io.logging import WithLoggerAndInterval
from hotspot3.connectors.bottleneck import BottleneckWrapper

from babachi.segmentation import GenomeSegmentator
from babachi.models import GenomeSNPsHandler, ChromosomeSNPsHandler


class BabachiWrapper(WithLoggerAndInterval):

    def craft_snps_collection(
            self,
            agg_cutcounts: ma.MaskedArray,
            per_window_trs: np.ndarray,
            fit_results: FitResults,
        ):
        step = self.config.babachi_segmentation_step

        bn_wrapper = self.copy_with_params(BottleneckWrapper)
        assumed_signal_mask = bn_wrapper.get_signal_mask_for_tr(agg_cutcounts, per_window_trs)
        background = agg_cutcounts.filled(np.nan)[::step]
        background[assumed_signal_mask[::step]] = np.nan
        starts = np.arange(0, len(background), dtype=np.uint32) * step
        valid_counts = ~np.isnan(background)
        
        chrom_handler = ChromosomeSNPsHandler(
            self.genomic_interval.chrom,
            positions=starts[valid_counts], 
            read_counts=np.stack(
                [
                    np.full(background.shape[0], fit_results.r, dtype=np.float32),
                    background
                ]
            ).T[valid_counts, :]
        )
        return GenomeSNPsHandler(chrom_handler)

    def run_segmentation(
            self,
            agg_cutcounts: ma.MaskedArray,
            per_window_trs: np.ndarray,
            fit_results: FitResults
        ):
        bads, chrom_bad = self.get_bads(fit_results)
        snps_collection = self.craft_snps_collection(
            agg_cutcounts,
            per_window_trs,
            fit_results
        )
        chrom_sizes = {self.genomic_interval.chrom: len(self.genomic_interval)}
        bad_segments = self.run_babachi(snps_collection, chrom_sizes, bads)
        if len(bad_segments) == 0:
            raise NotEnoughDataForContig
        return [
            GenomicInterval(x.chr, x.start, x.end, BAD=x.BAD / chrom_bad) for x in bad_segments
        ]
    
    def get_bads(self, fit_results: FitResults, max_bad=10):
        chrom_bad = fit_results.p / (1 - fit_results.p)
        mult = np.arange(1, max_bad + 0.1, 1)
        return [*(mult * chrom_bad), *(1 / mult[1:] * chrom_bad)], chrom_bad
    
    def run_babachi(self, snps_collection, chrom_sizes, bads):
        gs = GenomeSegmentator(
            snps_collection=snps_collection,
            chrom_sizes=chrom_sizes,
            jobs=1,
            logger_level=self.config.logger_level,
            segmentation_mode='binomial',
            states=bads,
            logger=self.logger,
            allele_reads_tr=0,
            b_penalty=self.config.babachi_boundary_penalty,
            min_seg_bp=self.config.babachi_min_segment_size,
            min_seg_snps=0,
            subchr_filter=self.config.babachi_min_segment_size // self.config.babachi_segmentation_step
        )
        return gs.estimate_BAD()

    def annotate_with_segments(self, shape, bad_segments: List[GenomicInterval]):
        babachi_result = np.zeros(shape, dtype=np.float16)
        for segment in bad_segments:
            babachi_result[segment.start:segment.end] = segment.BAD
        return babachi_result
