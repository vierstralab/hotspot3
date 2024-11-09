import dataclasses
import pandas as pd
import numpy as np


@dataclasses.dataclass
class GlobalFitResults:
    p: np.ndarray
    r: np.ndarray
    rmsea: np.ndarray
    fit_quantile: np.ndarray
    fit_threshold: np.ndarray


@dataclasses.dataclass
class WindowedFitResults:
    p: np.ndarray
    r: np.ndarray
    enough_bg_mask: np.ndarray


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



