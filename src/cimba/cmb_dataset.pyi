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
