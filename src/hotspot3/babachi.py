from hotspot3.models import GlobalFitResults, ProcessorConfig, WithLogger
from hotspot3.fit import WindowBackgroundFit
from babachi.segmentation import GenomeSegmentator
from babachi.models import GenomeSNPsHandler, ChromosomeSNPsHandler
import numpy as np
import numpy.ma as ma


class Segmentation(WithLogger):
    
    def run_babachi(self, agg_cutcounts: ma.MaskedArray, per_window_trs: np.ndarray, global_fit: GlobalFitResults, chrom_name, chrom_size):
        step = self.config.babachi_segmentation_step

        w_fit = WindowBackgroundFit(self.config, logger=self.logger)
        assumed_signal_mask = (agg_cutcounts >= per_window_trs).filled(False).astype(np.float32)
        assumed_signal_mask = w_fit.centered_running_nansum(
            assumed_signal_mask,
            window=self.config.window
        ) > 0
        background = agg_cutcounts.filled(np.nan)[::step]
        background[assumed_signal_mask[::step]] = np.nan
        starts = np.arange(0, len(background), dtype=np.uint32) * step

        bad = (1 - global_fit.p) / global_fit.p
        mult = np.linspace(1, 10, 20)
        bads = [*(mult * bad), *(1/mult[1:] * bad)]

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
            b_penalty=5
        )
        bad_segments = gs.estimate_BAD()
        gs.write_BAD(bad_segments, f"{self.chrom_name}.test.bed")

        return bad_segments