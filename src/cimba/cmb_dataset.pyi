"""Typed surface for Cimba's cmb_dataset module."""

from ._types import _Count
from .cmb_datasummary import DataSummary

class Dataset:
    """Resizable collection of unweighted float samples."""

    def __init__(self) -> None:
        """Create an empty dataset."""
        ...

    def add(self, value: float) -> _Count:
        """Add one sample and return the new sample count."""
        ...

    def values(self) -> list[float]:
        """Return the dataset's sample values in stored order."""
        ...

    def summary(self) -> DataSummary:
        """Compute and return an unweighted DataSummary for the dataset."""
        ...

    def reset(self) -> None:
        """Reset the dataset to an empty initialized state."""
        ...

    def copy(self) -> Dataset:
        """Return an owned copy of the dataset."""
        ...

    def merge(self, other: Dataset) -> Dataset:
        """Return a new dataset containing this dataset and other."""
        ...

    def sort(self) -> None:
        """Sort stored values in ascending order."""
        ...

    def acf(self, lags: int) -> list[float]:
        """Return autocorrelation coefficients from lag 0 through lags."""
        ...

    def pacf(self, lags: int) -> list[float]:
        """Return partial autocorrelation coefficients from lag 0 through lags."""
        ...

    @property
    def count(self) -> _Count:
        """Number of stored samples."""
        ...

    @property
    def min(self) -> float:
        """Smallest stored sample value."""
        ...

    @property
    def max(self) -> float:
        """Largest stored sample value."""
        ...

    @property
    def median(self) -> float:
        """Median of the stored samples."""
        ...

    def close(self) -> None:
        """Destroy the native dataset."""
        ...
