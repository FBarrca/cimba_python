"""Typed surface for Cimba's cmb_resource module."""

from ._types import _Amount, _ProcessSignal
from .cmb_process import Process
from .cmb_timeseries import TimeSeries

class Resource:
    """Binary semaphore resource that can be acquired, released, or preempted."""

    def __init__(self, name: str) -> None:
        """Create a named binary resource."""
        ...

    @property
    def name(self) -> str:
        """Resource name."""
        ...

    @property
    def in_use(self) -> _Amount:
        """1 if currently held, otherwise 0."""
        ...

    @property
    def available(self) -> _Amount:
        """1 if currently free, otherwise 0."""
        ...

    def acquire(self) -> _ProcessSignal:
        """Acquire the resource, waiting in priority order if it is unavailable."""
        ...

    def preempt(self) -> _ProcessSignal:
        """Acquire by preempting a lower-priority holder if possible."""
        ...

    def release(self) -> None:
        """Release the resource held by the current process."""
        ...

    def held_by(self, process: Process) -> _Amount:
        """Return 1 if process holds this resource, otherwise 0."""
        ...

    def start_recording(self) -> None:
        """Start recording binary resource usage history."""
        ...

    def stop_recording(self) -> None:
        """Stop recording binary resource usage history."""
        ...

    def history(self) -> TimeSeries:
        """Return an owned copy of the recorded resource usage history."""
        ...

    def close(self) -> None:
        """Destroy the native resource."""
        ...
