"""Typed surface for Cimba's cmb_buffer module."""

from typing import Final

from ._types import _Amount, _BufferGetResult, _BufferPutResult
from .cmb_timeseries import TimeSeries

UNLIMITED: Final[int]
"""Capacity sentinel for buffers and queues with no practical size limit."""

class Buffer:
    """Numeric fixed-capacity buffer with blocking put/get semantics."""

    def __init__(self, name: str, capacity: int | None = None) -> None:
        """Create a named buffer; capacity=None maps to Cimba's unlimited capacity."""
        ...

    @property
    def name(self) -> str:
        """Buffer name."""
        ...

    @property
    def capacity(self) -> _Amount:
        """Maximum buffer amount, or UNLIMITED."""
        ...

    @property
    def level(self) -> _Amount:
        """Current amount stored in the buffer."""
        ...

    @property
    def space(self) -> _Amount:
        """Current free capacity in the buffer."""
        ...

    def put(self, amount: _Amount = 1) -> _BufferPutResult:
        """Put amount into the buffer, waiting for space if needed.

        Returns (signal, remaining). On SUCCESS, remaining is zero. If interrupted,
        remaining is the amount not yet placed.
        """
        ...

    def get(self, amount: _Amount = 1) -> _BufferGetResult:
        """Get amount from the buffer, waiting for content if needed.

        Returns (signal, obtained). On SUCCESS, obtained equals the requested
        amount. If interrupted, obtained is the partial amount collected.
        """
        ...

    def start_recording(self) -> None:
        """Start recording the buffer level time series."""
        ...

    def stop_recording(self) -> None:
        """Stop recording the buffer level time series."""
        ...

    def history(self) -> TimeSeries:
        """Return an owned copy of the recorded buffer level history."""
        ...

    def close(self) -> None:
        """Destroy the native buffer."""
        ...
