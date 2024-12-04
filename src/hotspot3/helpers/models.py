import dataclasses
import pandas as pd
import numpy as np
from typing import List, Dict
from genome_tools import GenomicInterval


@dataclasses.dataclass
class FitResults:
    p: np.ndarray = np.nan
    r: np.ndarray = np.nan
    rmsea: np.ndarray = np.nan
    fit_quantile: np.ndarray = np.nan
    fit_threshold: np.ndarray = np.nan
    n_signal: int = np.nan
    n_total: int = np.nan


@dataclasses.dataclass
class WindowedFitResults:
    p: np.ndarray
    r: np.ndarray
    enough_bg_mask: np.ndarray


@dataclasses.dataclass
class DataForFit:
    bin_edges: np.ndarray
    value_counts: np.ndarray
    n_signal_bins: int
    agg_cutcounts: np.ndarray
    max_counts_with_flanks: np.ndarray

@dataclasses.dataclass
class SPOTEstimationData:
    total_tags: np.ndarray
    total_tags_background: np.ndarray
    segment_spot_scores: np.ndarray
    total_bases: np.ndarray
    valid_scores: np.ndarray

@dataclasses.dataclass
class SPOTEstimationResults:
    spot_score: float
    spot_score_std: float
    slope: float
    intercept: float
    inliers_mask: np.ndarray
    outlier_distance: np.ndarray
    min_bg_tags_fraction: np.ndarray


@dataclasses.dataclass
class FDRData:
    potentially_signif_pvals: np.ndarray
    n_tests: int

@dataclasses.dataclass
class SampleFDRdata(FDRData):
    sample_id: str
    chrom_pos_mapping: List[GenomicInterval]


@dataclasses.dataclass
class MultiSampleFDRData(FDRData):
    sample_id_correspondance: pd.DataFrame
    chrom_pos_mappings: Dict[List[GenomicInterval]]


@dataclasses.dataclass
class ProcessorOutputData:
    """
    Dataclass for storing the output of ChromosomeProcessor and GenomeProcessor methods.
    """
    id: str
    data_df: pd.DataFrame

class NotEnoughDataForContig(Exception):
    """Exception raised when a required contig is not present."""

    def __init__(self, message="No contig is present in the provided data."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"



