import dataclasses
import pandas as pd
import logging
import numpy as np


@dataclasses.dataclass
class ProcessorConfig:
    """
    Parameters:
        - window: Bandwidth for signal smoothing.
        - min_mappable: Minimum number of mappable bases for a window to be tested.
    
        - bg_window: Window size for aggregating background cutcounts.
        - min_mappable_bg: Minimum number of mappable bases for a window to be considered in background.

        - density_step: Step size for extracting density.
        - min_hotspot_width: Minimum width for a region to be called a hotspot.
    
        - signal_tr: Quantile threshold for outlier detection for background distribution fit.
        - adaptive_signal_tr: Use adaptive signal threshold. Signal_tr is ignored if True.

        - nonzero_window_fit: Minimum fraction of nonzero values in a background window to fit the model.
        - fdr_method: Method for FDR calculation. 'bh and 'by' are supported. 'bh' (default) is tested.
        - cpus: Number of CPUs to use. Won't use more than the number of chromosomes.

        - save_debug: Save debug information.
        - modwt_level: Level of MODWT decomposition. 7 is tested.
        - logger_level: Logging level.
    """
    window: int = 151
    bg_window: int = 50001
    min_mappable_bg_frac: float = 0.2
    density_step: int = 20
    max_background_prop: float = 0.99
    min_background_prop: float = 0.6

    signal_prop_sampling_step: int = 75
    signal_prop_interpolation_step: int = 500 # shouldn't be less than signal_prop_n_samples!!!!!
    num_background_bins: int = 20
    num_signal_bins: int = 100
    rmsea_tr: float = 0.05
    nonzero_windows_to_fit: float = 0.01
    outlier_detection_tr: float = 0.99
    fdr_method: str = 'by'
    cpus: int = 1
    save_debug: bool = False
    modwt_level: int = 7
    logger_level: int = logging.INFO


@dataclasses.dataclass
class ProcessorOutputData:
    """
    Dataclass for storing the output of ChromosomeProcessor and GenomeProcessor methods.
    """
    identificator: str
    data_df: pd.DataFrame


@dataclasses.dataclass
class FitResults:
    p: np.ndarray
    r: np.ndarray
    rmsea: np.ndarray
    fit_quantile: np.ndarray
    fit_threshold: np.ndarray
    enough_bg_mask: np.ndarray = None


class NoContigPresentError(Exception):
    """Exception raised when a required contig is not present."""

    def __init__(self, message="No contig is present in the provided data."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"
