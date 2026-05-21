"""Typed surface for Cimba's cmb_objectqueue module."""

from ._types import _Count, _ObjectQueueGetResult, _ProcessSignal
from .cmb_timeseries import TimeSeries

class ObjectQueue:
    """FIFO queue for Python objects with blocking put/get semantics."""

    def __init__(self, name: str, capacity: int | None = None) -> None:
        """Create a named FIFO object queue; capacity=None means unlimited."""
        ...

    @property
    def name(self) -> str:
        """Queue name."""
        ...

    @property
    def capacity(self) -> _Count:
        """Maximum queue length, or UNLIMITED."""
        ...

    @property
    def length(self) -> _Count:
        """Current number of queued objects."""
        ...

    @property
    def space(self) -> _Count:
        """Current number of available queue slots."""
        ...

    def put(self, obj: object) -> _ProcessSignal:
        """Put one Python object into the queue, waiting for space if needed."""
        ...

    def get(self) -> _ObjectQueueGetResult:
        """Get the next object in FIFO order; returns (signal, object_or_none)."""
        ...

    def position(self, obj: object) -> _Count:
        """Return the object's 1-based queue position, or 0 if not present."""
        ...

    def start_recording(self) -> None:
        """Start recording queue length history."""
        ...

    def stop_recording(self) -> None:
        """Stop recording queue length history."""
        ...

    def history(self) -> TimeSeries:
        """Return an owned copy of the recorded queue length history."""
        ...

    def close(self) -> None:
        """Destroy the native queue and release queued Python object references."""
        ...
