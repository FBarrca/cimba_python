"""Typed surface for Cimba's cmb_priorityqueue module."""

from ._types import (
    _Count,
    _Priority,
    _PriorityQueueGetResult,
    _PriorityQueuePutResult,
    _QueueHandle,
)
from .cmb_timeseries import TimeSeries

class PriorityQueue:
    """Priority queue for Python objects with blocking put/get semantics."""

    def __init__(self, name: str, capacity: int | None = None) -> None:
        """Create a named priority queue; capacity=None means unlimited."""
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

    def put(self, obj: object, priority: _Priority = 0) -> _PriorityQueuePutResult:
        """Put an object with priority; higher priority is retrieved first.

        Returns (signal, handle). The handle can be used to locate, cancel, or
        reprioritize the item while it remains queued.
        """
        ...

    def get(self) -> _PriorityQueueGetResult:
        """Get the highest-priority object; returns (signal, object_or_none)."""
        ...

    def position(self, handle: _QueueHandle) -> _Count:
        """Return the item's 1-based priority position, or 0 if not queued."""
        ...

    def cancel(self, handle: _QueueHandle) -> bool:
        """Remove an item by handle and release its Python reference."""
        ...

    def reprioritize(self, handle: _QueueHandle, priority: _Priority) -> None:
        """Change an item's priority and reshuffle the queue."""
        ...

    def start_recording(self) -> None:
        """Start recording priority queue length history."""
        ...

    def stop_recording(self) -> None:
        """Stop recording priority queue length history."""
        ...

    def history(self) -> TimeSeries:
        """Return an owned copy of the recorded queue length history."""
        ...

    def close(self) -> None:
        """Destroy the native priority queue and release queued object references."""
        ...
