"""Typed surface for Cimba's cmb_dataset module."""

from os import PathLike

from ._types import _Count
from .cmb_datasummary import DataSummary

_Path = str | bytes | PathLike[str] | PathLike[bytes]


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
    def mean(self) -> float:
        """Mean of the stored samples."""
        ...

    @property
    def std(self) -> float:
        """Sample standard deviation of the stored samples."""
        ...

    @property
    def stddev(self) -> float:
        """Sample standard deviation of the stored samples."""
        ...

    @property
    def median(self) -> float:
        """Median of the stored samples."""
        ...

    def quantile(self, q: float) -> float:
        """Return quantile q in [0, 1] using linear interpolation."""
        ...

    def print(self) -> int:
        """Print raw dataset values to stdout."""
        ...

    def print_file(self, path: _Path, append: bool = True) -> int:
        """Write raw dataset values to path."""
        ...

    def fivenum(self) -> int:
        """Print the native five-number summary to stdout."""
        ...

    def fivenum_file(self, path: _Path, append: bool = True) -> int:
        """Write the native five-number summary to path."""
        ...

    def histogram(self, bins: int = 20, low: float = 0.0,
                  high: float = 0.0) -> int:
        """Print the native text histogram to stdout."""
        ...

    def histogram_file(self, path: _Path, append: bool = True, bins: int = 20,
                       low: float = 0.0, high: float = 0.0) -> int:
        """Write the native text histogram to path."""
        ...

    def correlogram(self, lags: int = 20) -> int:
        """Print the native ACF correlogram to stdout."""
        ...

    def correlogram_file(self, path: _Path, append: bool = True,
                         lags: int = 20) -> int:
        """Write the native ACF correlogram to path."""
        ...

    def pacf_correlogram(self, lags: int = 20) -> int:
        """Print the native PACF correlogram to stdout."""
        ...

    def pacf_correlogram_file(self, path: _Path, append: bool = True,
                              lags: int = 20) -> int:
        """Write the native PACF correlogram to path."""
        ...

    def close(self) -> None:
        """Destroy the native dataset."""
        ...
