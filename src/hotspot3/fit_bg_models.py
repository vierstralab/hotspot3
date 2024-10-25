import numpy as np
import numpy.ma as ma
import logging
from hotspot3.models import NoContigPresentError, ProcessorConfig


class BackgroundFit:
    def __init__(self, chrom_name, config: ProcessorConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.chrom_name = chrom_name

    def fit_windowed_model(self):
        ...
    
    def fit_global_model(self, agg_cutcounts: ma.MaskedArray, high_signal_mask):
        agg_cutcounts = agg_cutcounts[~high_signal_mask]
        has_enough_background = ma.sum(agg_cutcounts > 0) / agg_cutcounts.count() > self.config.nonzero_windows_to_fit
        if not has_enough_background:
            self.logger.warning(f"Not enough background signal for global fit of {self.chrom_name}. Skipping...")
            raise NoContigPresentError
        
        mean = ma.mean(agg_cutcounts)
        variance = ma.var(agg_cutcounts, mean=mean, ddof=1)
        return mean, variance
