"""Typed surface for Cimba's cmb_resourcepool module."""

from ._types import _Amount, _ProcessSignal
from .cmb_process import Process
from .cmb_timeseries import TimeSeries

class ResourcePool:
    """Counting semaphore resource pool with acquire, release, and preempt."""

    def __init__(self, name: str, capacity: _Amount) -> None:
        """Create a named resource pool with a positive integer capacity."""
        ...

    @property
    def name(self) -> str:
        """Resource pool name."""
        ...

    @property
    def capacity(self) -> _Amount:
        """Maximum amount assignable across all processes."""
        ...

    @property
    def in_use(self) -> _Amount:
        """Amount currently held by all processes."""
        ...

    @property
    def available(self) -> _Amount:
        """Amount currently free for acquisition."""
        ...

    def acquire(self, amount: _Amount = 1) -> _ProcessSignal:
        """Acquire amount from the pool, waiting for availability if needed."""
        ...

    def preempt(self, amount: _Amount = 1) -> _ProcessSignal:
        """Acquire amount, taking from lower-priority holders if possible."""
        ...

    def release(self, amount: _Amount = 1) -> None:
        """Release amount held by the current process back to the pool."""
        ...

    def held_by(self, process: Process) -> _Amount:
        """Return how much of this pool is currently held by process."""
        ...

    def start_recording(self) -> None:
        """Start recording pool usage history."""
        ...

    def stop_recording(self) -> None:
        """Stop recording pool usage history."""
        ...

    def history(self) -> TimeSeries:
        """Return an owned copy of the recorded pool usage history."""
        ...

    def close(self) -> None:
        """Destroy the native resource pool."""
        ...
