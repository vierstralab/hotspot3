import dataclasses
import logging


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
    min_mappable_bases_proportion: float = 0.05
    
    # Signal smoothing
    filter:str = 'haar'
    modwt_level: int = 7

    # Background model
    bg_window: int = 50001
    exclude_peak_flank_length: int = 0 # half window
    min_background_prop: float = 0.6
    max_background_prop: float = 0.99

    signal_prop_sampling_step: int = 75
    signal_prop_interpolation_step: int = 1500 # shouldn't be less than signal_prop_n_samples!!!!!

    # RMSEA calculation
    num_background_bins: int = 20
    num_signal_bins: int = 100
    rmsea_tr: float = 0.05
    min_obs_rmsea: int = 5

    # Segmentation
    babachi_segmentation_step: int = 500
    babachi_boundary_penalty: int = 9
    babachi_min_segment_size: int = 5000

    fdr_method: str = 'by'
    density_step: int = 20

    cpus: int = 1
    tmp_dir: str = None
    save_debug: bool = False
    logger_level: int = logging.INFO