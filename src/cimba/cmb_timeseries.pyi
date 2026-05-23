"""Typed surface for Cimba's cmb_timeseries module."""

from ._types import _Count, _TimeSeriesRow
from .cmb_wtdsummary import WeightedSummary

class TimeSeries:
    """Resizable sequence of (time, value, weight) samples."""

    def __init__(self) -> None:
        """Create an empty time series."""
        ...

    def add(self, value: float, time: float) -> _Count:
        """Add a value at a simulation timestamp and return the new count."""
        ...

    def finalize(self, time: float) -> _Count:
        """Close the last interval by repeating the last value at time."""
        ...

    def values(self) -> list[_TimeSeriesRow]:
        """Return rows as (time, value, weight) tuples."""
        ...

    def summary(self) -> WeightedSummary:
        """Compute a duration-weighted summary of the time series."""
        ...

    def reset(self) -> None:
        """Reset the time series to an empty initialized state."""
        ...

    def copy(self) -> TimeSeries:
        """Return an owned copy of the time series."""
        ...

    def sort_by_value(self) -> None:
        """Sort rows by sample value."""
        ...

    def sort_by_time(self) -> None:
        """Sort rows by timestamp."""
        ...

    def acf(self, lags: int) -> list[float]:
        """Return autocorrelation coefficients from lag 0 through lags."""
        ...

    def pacf(self, lags: int) -> list[float]:
        """Return partial autocorrelation coefficients from lag 0 through lags."""
        ...

    @property
    def count(self) -> _Count:
        """Number of time-stamped rows."""
        ...

    @property
    def min(self) -> float:
        """Smallest sample value."""
        ...

    @property
    def max(self) -> float:
        """Largest sample value."""
        ...

    @property
    def median(self) -> float:
        """Duration-weighted median sample value."""
        ...

    def close(self) -> None:
        """Destroy the native time series."""
        ...
