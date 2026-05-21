"""Typed surface for Cimba's cmb_wtdsummary module."""

from ._types import _Count

class WeightedSummary:
    """Single-pass duration/weight-aware summary of sample moments."""

    def __init__(self) -> None:
        """Create an empty weighted summary."""
        ...

    def add(self, value: float, weight: float = 1.0) -> _Count:
        """Add one weighted sample and return the new sample count."""
        ...

    @property
    def count(self) -> _Count:
        """Number of weighted samples summarized."""
        ...

    @property
    def weight_sum(self) -> float:
        """Total accumulated weight."""
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
        """Weighted mean."""
        ...

    @property
    def variance(self) -> float:
        """Weighted variance."""
        ...

    @property
    def stddev(self) -> float:
        """Weighted standard deviation."""
        ...

    @property
    def skewness(self) -> float:
        """Weighted skewness."""
        ...

    @property
    def kurtosis(self) -> float:
        """Weighted excess kurtosis."""
        ...

    def close(self) -> None:
        """Destroy the native weighted summary."""
        ...
