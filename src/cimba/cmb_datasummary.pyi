"""Typed surface for Cimba's cmb_datasummary module."""

from ._types import _Count

class DataSummary:
    """Single-pass unweighted summary of sample moments."""

    def __init__(self) -> None:
        """Create an empty data summary."""
        ...

    def add(self, value: float) -> _Count:
        """Add one sample value and return the new sample count."""
        ...

    @property
    def count(self) -> _Count:
        """Number of samples summarized."""
        ...

    @property
    def min(self) -> float:
        """Smallest sample value seen."""
        ...

    @property
    def max(self) -> float:
        """Largest sample value seen."""
        ...

    @property
    def mean(self) -> float:
        """Sample mean."""
        ...

    @property
    def variance(self) -> float:
        """Sample variance."""
        ...

    @property
    def stddev(self) -> float:
        """Sample standard deviation."""
        ...

    @property
    def skewness(self) -> float:
        """Sample skewness."""
        ...

    @property
    def kurtosis(self) -> float:
        """Sample excess kurtosis."""
        ...

    def close(self) -> None:
        """Destroy the native data summary."""
        ...
