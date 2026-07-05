"""Grouped dataset helpers for the ``cimba.sim`` API."""

from typing import TYPE_CHECKING

import numpy as _np

from numba import njit

from .. import _bindings as _b
from .._declarations import Handle

if TYPE_CHECKING:
    def add(dataset: Handle, value: float) -> int:
        """Record an observation; returns the observation count."""
        ...

    def mean(dataset: Handle) -> float:
        """Mean of the observations tallied so far."""
        ...

    def count(dataset: Handle) -> int:
        """Number of observations tallied so far."""
        ...

    def min(dataset: Handle) -> float:
        """Smallest observation tallied so far."""
        ...

    def max(dataset: Handle) -> float:
        """Largest observation tallied so far."""
        ...

    def std(dataset: Handle) -> float:
        """Sample standard deviation of the observations (0 if < 2)."""
        ...

    def median(dataset: Handle) -> float:
        """Median of the observations tallied so far (0 if empty)."""
        ...

    def quantile(dataset: Handle, q: float) -> float:
        """Quantile q in [0, 1] by linear interpolation over sorted values."""
        ...

    def print_file(dataset: Handle, path: Handle, append: int = 1) -> int:
        """Write raw dataset values to path."""
        ...

    def print(dataset: Handle) -> int:
        """Print raw dataset values to stdout."""
        ...

    def fivenum_file(dataset: Handle, path: Handle, append: int = 1) -> int:
        """Write the native dataset five-number summary to path."""
        ...

    def fivenum(dataset: Handle) -> int:
        """Print the native dataset five-number summary to stdout."""
        ...

    def histogram_file(dataset: Handle, path: Handle, append: int = 1,
                       bins: int = 20, low: float = 0.0,
                       high: float = 0.0) -> int:
        """Write the native dataset text histogram to path."""
        ...

    def histogram(dataset: Handle, bins: int = 20, low: float = 0.0,
                  high: float = 0.0) -> int:
        """Print the native dataset text histogram to stdout."""
        ...

    def correlogram_file(dataset: Handle, path: Handle, append: int = 1,
                         lags: int = 20) -> int:
        """Write the native dataset ACF correlogram to path."""
        ...

    def correlogram(dataset: Handle, lags: int = 20) -> int:
        """Print the native dataset ACF correlogram to stdout."""
        ...

    def pacf_correlogram_file(dataset: Handle, path: Handle, append: int = 1,
                              lags: int = 20) -> int:
        """Write the native dataset PACF correlogram to path."""
        ...

    def pacf_correlogram(dataset: Handle, lags: int = 20) -> int:
        """Print the native dataset PACF correlogram to stdout."""
        ...

else:
    add = _b.dataset_add
    mean = _b.dataset_mean
    count = _b.dataset_count
    min = _b.dataset_min
    max = _b.dataset_max
    std = _b.dataset_std
    median = _b.dataset_median
    quantile = _b.dataset_quantile
    print_file = _b.dataset_print_file
    fivenum_file = _b.dataset_fivenum_file
    histogram_file = _b.dataset_histogram_file
    correlogram_file = _b.dataset_correlogram_file
    pacf_correlogram_file = _b.dataset_pacf_correlogram_file

    @njit
    def print(dataset):
        return print_file(dataset, 0, _np.uint64(1))

    @njit
    def fivenum(dataset):
        return fivenum_file(dataset, 0, _np.uint64(1))

    @njit
    def histogram(dataset, bins=20, low=0.0, high=0.0):
        return histogram_file(dataset, 0, _np.uint64(1),
                              _np.uint64(bins), low, high)

    @njit
    def correlogram(dataset, lags=20):
        return correlogram_file(dataset, 0, _np.uint64(1),
                                _np.uint64(lags))

    @njit
    def pacf_correlogram(dataset, lags=20):
        return pacf_correlogram_file(dataset, 0, _np.uint64(1),
                                     _np.uint64(lags))


__all__ = [
    "add",
    "mean",
    "count",
    "min",
    "max",
    "std",
    "median",
    "quantile",
    "print",
    "print_file",
    "fivenum",
    "fivenum_file",
    "histogram",
    "histogram_file",
    "correlogram",
    "correlogram_file",
    "pacf_correlogram",
    "pacf_correlogram_file",
]
