"""Typed surface for Cimba's cmb_event module."""

from collections.abc import Callable
from types import TracebackType
from typing import Self

from ._types import _Count, _EventHandle, _Priority

def time() -> float:
    """Return the current simulation clock time for the active event queue."""
    ...

class Simulation:
    """Own the thread-local Cimba event queue, simulation clock, and PRNG state.

    Use as a context manager. Cimba objects created while a simulation is active
    are kept alive by the simulation and closed in reverse creation order.
    """

    seed_used: int
    """Seed used to initialize the simulation's thread-local PRNG."""

    def __init__(
        self,
        start_time: float = 0.0,
        seed: int | None = None,
        log_info: bool = False,
    ) -> None:
        """Initialize a simulation starting at start_time with an optional seed."""
        ...

    def __enter__(self) -> Self:
        """Enter the simulation context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        """Close the simulation context and propagate any exception."""
        ...

    @property
    def closed(self) -> bool:
        """Whether this simulation has already released its native state."""
        ...

    @property
    def now(self) -> float:
        """Current simulation time."""
        ...

    @property
    def event_count(self) -> _Count:
        """Number of scheduled future events in the event queue."""
        ...

    @property
    def current_event(self) -> _EventHandle:
        """Handle of the currently or most recently executed event."""
        ...

    def stop_at(self, when: float, priority: _Priority = 0) -> _EventHandle:
        """Schedule a stop event at absolute simulation time when."""
        ...

    def schedule(
        self,
        callback: Callable[[object | None, object | None], object],
        when: float,
        subject: object | None = None,
        obj: object | None = None,
        priority: _Priority = 0,
    ) -> _EventHandle:
        """Schedule a Python callback at absolute simulation time when."""
        ...

    def schedule_native(
        self,
        action_capsule: object,
        when: float,
        subject_capsule: object | None = None,
        object_capsule: object | None = None,
        priority: _Priority = 0,
    ) -> _EventHandle:
        """Schedule a native Cimba event function capsule."""
        ...

    def cancel_event(self, handle: _EventHandle) -> bool:
        """Cancel a scheduled event by handle; return True when it was found."""
        ...

    def reschedule_event(self, handle: _EventHandle, when: float) -> bool:
        """Move a scheduled event to another absolute simulation time."""
        ...

    def reprioritize_event(self, handle: _EventHandle, priority: _Priority) -> bool:
        """Change the priority of a scheduled event."""
        ...

    def is_event_scheduled(self, handle: _EventHandle) -> bool:
        """Return whether the event handle is still scheduled."""
        ...

    def event_time(self, handle: _EventHandle) -> float:
        """Return the scheduled absolute time for an event."""
        ...

    def event_priority(self, handle: _EventHandle) -> _Priority:
        """Return the scheduled priority for an event."""
        ...

    def clear(self) -> None:
        """Clear all scheduled events, ending the current run."""
        ...

    def execute_next(self) -> bool:
        """Execute the next scheduled event; return False if the queue is empty."""
        ...

    def execute(self) -> None:
        """Run scheduled events until the event queue is empty."""
        ...

    def close(self) -> None:
        """Stop owned processes and release Cimba's thread-local state."""
        ...
