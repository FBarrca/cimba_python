"""Typed surface for Cimba's cmb_process module."""

from typing import Final, NoReturn, Self

from ._types import (
    _EventHandle,
    _Priority,
    _ProcessFunc,
    _ProcessSignal,
    _ProcessStatus,
    _TimerHandle,
)

SUCCESS: Final[int]
"""Process call returned normally."""
PREEMPTED: Final[int]
"""Process lost a held resource to a higher-priority process."""
INTERRUPTED: Final[int]
"""Generic interrupt signal sent to a waiting or yielded process."""
STOPPED: Final[int]
"""Process was stopped while another process was waiting for it."""
CANCELLED: Final[int]
"""Pending resource-style request was cancelled."""
TIMEOUT: Final[int]
"""Default signal for process timer wakeups."""

PROCESS_CREATED: Final[int]
"""Process has been initialized but has not started running yet."""
PROCESS_RUNNING: Final[int]
"""Process is active or waiting inside the simulation."""
PROCESS_FINISHED: Final[int]
"""Process has returned, exited, or been stopped."""

def hold(duration: float) -> _ProcessSignal:
    """Suspend the current process for duration simulated time units."""
    ...

def yield_process() -> _ProcessSignal:
    """Yield the current process until another process, timer, or event resumes it."""
    ...

def wait_event(handle: _EventHandle) -> _ProcessSignal:
    """Yield the current process until a scheduled event fires or is canceled."""
    ...

def process_exit(value: object | None = None) -> NoReturn:
    """Exit the current process immediately with an optional Python exit value."""
    ...

def current_process() -> Process | None:
    """Return the currently running process, or None outside process execution."""
    ...

class Process:
    """Named stackful Cimba process executing a Python callable.

    The callable receives the Process itself and the context object. It may call
    hold(), yield_process(), queue/resource operations, or return an exit value.
    """

    def __init__(
        self,
        name: str,
        func: _ProcessFunc,
        context: object | None = None,
        priority: _Priority = 0,
    ) -> None:
        """Create a process with a name, function, context, and initial priority."""
        ...

    @property
    def name(self) -> str:
        """Process name as seen by Cimba logging and diagnostics."""
        ...

    @property
    def priority(self) -> _Priority:
        """Current process priority used in waiting/resource queues."""
        ...

    @priority.setter
    def priority(self, value: _Priority) -> None:
        """Set the process priority and let Cimba reorder wait queues as needed."""
        ...

    @property
    def status(self) -> _ProcessStatus:
        """Current process lifecycle state."""
        ...

    def start(self) -> Self:
        """Schedule the process to start at the current simulation time."""
        ...

    def stop(self) -> _ProcessSignal:
        """Request cooperative cancellation of a running Python-backed process."""
        ...

    def interrupt(self, signal: _ProcessSignal = INTERRUPTED, priority: _Priority = 0) -> None:
        """Interrupt a waiting process with a non-success signal."""
        ...

    def resume(self, signal: _ProcessSignal = SUCCESS) -> None:
        """Schedule a yielded process to resume with the given signal."""
        ...

    def wait(self) -> _ProcessSignal:
        """Wait until this process finishes, returning the wakeup signal."""
        ...

    def timer_add(self, duration: float, signal: _ProcessSignal = TIMEOUT) -> _TimerHandle:
        """Add an independent timer that resumes this process after duration."""
        ...

    def timer_set(self, duration: float, signal: _ProcessSignal = TIMEOUT) -> _TimerHandle:
        """Clear existing timers and set one timer for this process."""
        ...

    def timer_cancel(self, handle: _TimerHandle) -> bool:
        """Cancel a timer by handle; return True when it was found."""
        ...

    def timers_clear(self) -> None:
        """Cancel all timers currently scheduled for this process."""
        ...

    def exit_value(self) -> object | None:
        """Return the Python value produced by a finished process, if any."""
        ...

    def close(self) -> None:
        """Release the native process and any owned Python exit value."""
        ...
