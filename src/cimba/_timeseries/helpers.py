"""Grouped timeseries helpers for the ``cimba.sim`` API."""

from typing import TYPE_CHECKING

import numpy as _np

from numba import njit

from .. import _bindings as _b
from .._declarations import Handle

if TYPE_CHECKING:
    def queue_history(queue: Handle) -> Handle:
        """Native timeseries history for a queue."""
        ...

    def resource_history(resource: Handle) -> Handle:
        """Native timeseries history for a resource."""
        ...

    def pool_history(pool: Handle) -> Handle:
        """Native timeseries history for a resource pool."""
        ...

    def store_history(store: Handle) -> Handle:
        """Native timeseries history for a store/object queue."""
        ...

    def pq_history(pqueue: Handle) -> Handle:
        """Native timeseries history for a priority queue."""
        ...

    def count(timeseries: Handle) -> int:
        """Number of samples tallied so far."""
        ...

    def mean(timeseries: Handle) -> float:
        """Time-weighted mean of the samples tallied so far."""
        ...

    def min(timeseries: Handle) -> float:
        """Smallest sample value tallied so far."""
        ...

    def max(timeseries: Handle) -> float:
        """Largest sample value tallied so far."""
        ...

    def std(timeseries: Handle) -> float:
        """Time-weighted sample standard deviation."""
        ...

    def median(timeseries: Handle) -> float:
        """Time-weighted median."""
        ...

    def print_file(timeseries: Handle, path: Handle, append: int = 1) -> int:
        """Write raw timeseries rows to path."""
        ...

    def print(timeseries: Handle) -> int:
        """Print raw timeseries rows to stdout."""
        ...

    def fivenum_file(timeseries: Handle, path: Handle, append: int = 1) -> int:
        """Write the native weighted five-number summary to path."""
        ...

    def fivenum(timeseries: Handle) -> int:
        """Print the native weighted five-number summary to stdout."""
        ...

    def histogram_file(timeseries: Handle, path: Handle, append: int = 1,
                       bins: int = 20, low: float = 0.0,
                       high: float = 0.0) -> int:
        """Write the native weighted text histogram to path."""
        ...

    def histogram(timeseries: Handle, bins: int = 20, low: float = 0.0,
                 high: float = 0.0) -> int:
        """Print the native weighted text histogram to stdout."""
        ...

    def correlogram_file(timeseries: Handle, path: Handle, append: int = 1,
                         lags: int = 20) -> int:
        """Write the native timeseries ACF correlogram to path."""
        ...

    def correlogram(timeseries: Handle, lags: int = 20) -> int:
        """Print the native timeseries ACF correlogram to stdout."""
        ...

    def pacf_correlogram_file(timeseries: Handle, path: Handle,
                              append: int = 1, lags: int = 20) -> int:
        """Write the native timeseries PACF correlogram to path."""
        ...

    def pacf_correlogram(timeseries: Handle, lags: int = 20) -> int:
        """Print the native timeseries PACF correlogram to stdout."""
        ...

else:
    queue_history = _b.buffer_history
    resource_history = _b.resource_history
    pool_history = _b.resourcepool_history
    store_history = _b.objectqueue_history
    pq_history = _b.priorityqueue_history

    count = _b.timeseries_count
    mean = _b.timeseries_mean
    min = _b.timeseries_min
    max = _b.timeseries_max
    std = _b.timeseries_std
    median = _b.timeseries_median
    print_file = _b.timeseries_print_file
    fivenum_file = _b.timeseries_fivenum_file
    histogram_file = _b.timeseries_histogram_file
    correlogram_file = _b.timeseries_correlogram_file
    pacf_correlogram_file = _b.timeseries_pacf_correlogram_file

    @njit
    def print(timeseries):
        return print_file(timeseries, 0, _np.uint64(1))

    @njit
    def fivenum(timeseries):
        return fivenum_file(timeseries, 0, _np.uint64(1))

    @njit
    def histogram(timeseries, bins=20, low=0.0, high=0.0):
        return histogram_file(timeseries, 0, _np.uint64(1),
                              _np.uint64(bins), low, high)

    @njit
    def correlogram(timeseries, lags=20):
        return correlogram_file(timeseries, 0, _np.uint64(1),
                                _np.uint64(lags))

    @njit
    def pacf_correlogram(timeseries, lags=20):
        return pacf_correlogram_file(timeseries, 0, _np.uint64(1),
                                     _np.uint64(lags))


#: entity-kind binding name -> history getter, keyed the same way
#: ``_FieldKind.binding`` is on queue/resource/pool/store/pqueues fields.
HISTORY_GETTERS = {
    "buffer": queue_history,
    "resource": resource_history,
    "resourcepool": pool_history,
    "objectqueue": store_history,
    "priorityqueue": pq_history,
}

__all__ = [
    "queue_history",
    "resource_history",
    "pool_history",
    "store_history",
    "pq_history",
    "count",
    "mean",
    "min",
    "max",
    "std",
    "median",
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
    "HISTORY_GETTERS",
]
