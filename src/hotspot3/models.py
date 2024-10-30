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
    min_mappable: int = 76
    bg_window: int = 50001
    min_mappable_bg: int = 10000
    density_step: int = 20
    max_background_prop: float = 0.99
    min_background_prop: float = 0.75

    signal_prop_n_samples: int = 667 # out of bg_window
    signal_prop_step: int = 1500 # shouldn't be less than signal_prop_n_samples!!!!!
    num_background_bins: int = 20
    num_signal_bins: int = 100
    rmsea_tr: float = 0.05
    adaptive_signal_tr: bool = False
    nonzero_windows_to_fit: float = 0.01
    fdr_method: str = 'bh'
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
    poisson_fit_params: np.ndarray = None


class NoContigPresentError(Exception):
    """Exception raised when a required contig is not present."""

    def __init__(self, message="No contig is present in the provided data."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"
