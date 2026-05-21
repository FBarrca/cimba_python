"""Typed Python surface for the Cimba discrete-event simulation API.

The runtime implementation lives in the Cython extension. These stubs add
Python-facing type signatures and IDE hover documentation for the Cimba concepts:
process signals, event/timer/queue handles, resources, queues, random variates,
and summary collectors.
"""

from collections.abc import Callable
from types import TracebackType
from typing import Any, Final, NoReturn, Self, TypeAlias

_Amount: TypeAlias = int
_Count: TypeAlias = int
_EventHandle: TypeAlias = int
_LoggerFlags: TypeAlias = int
_Priority: TypeAlias = int
_ProcessSignal: TypeAlias = int
_ProcessStatus: TypeAlias = int
_QueueHandle: TypeAlias = int
_TimerHandle: TypeAlias = int

_ProcessFunc: TypeAlias = Callable[["Process", Any], object]
_ConditionPredicate: TypeAlias = Callable[["Process | None", Any], bool]

_BufferPutResult: TypeAlias = tuple[_ProcessSignal, _Amount]
_BufferGetResult: TypeAlias = tuple[_ProcessSignal, _Amount]
_ObjectQueueGetResult: TypeAlias = tuple[_ProcessSignal, object | None]
_PriorityQueuePutResult: TypeAlias = tuple[_ProcessSignal, _QueueHandle]
_PriorityQueueGetResult: TypeAlias = tuple[_ProcessSignal, object | None]
_TimeSeriesRow: TypeAlias = tuple[float, float, float]

UNLIMITED: Final[int]
"""Capacity sentinel for buffers and queues with no practical size limit."""

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

LOGGER_FATAL: Final[int]
"""Fatal logger flag; fatal messages abort the program."""
LOGGER_ERROR: Final[int]
"""Error logger flag."""
LOGGER_WARNING: Final[int]
"""Warning logger flag."""
LOGGER_INFO: Final[int]
"""Internal informational logger flag."""

def native_version() -> str:
    """Return the version string reported by the bundled Cimba C library."""
    ...

def logger_flags_on(flags: _LoggerFlags) -> None:
    """Enable one or more Cimba logger flags in the current thread."""
    ...

def logger_flags_off(flags: _LoggerFlags) -> None:
    """Disable one or more Cimba logger flags in the current thread."""
    ...

def hwseed() -> int:
    """Return a hardware-derived 64-bit seed suitable for Cimba's PRNG."""
    ...

def seed(value: int | None = None) -> int:
    """Initialize the thread-local PRNG and return the seed that was used."""
    ...

def current_seed() -> int:
    """Return the seed used for the current thread's PRNG stream."""
    ...

def random() -> float:
    """Draw a continuous uniform random variate on [0.0, 1.0]."""
    ...

def random_u64() -> int:
    """Draw a raw uniformly distributed 64-bit pseudo-random bit pattern."""
    ...

def fmix64(seed: int, nonce: int) -> int:
    """Mix a master seed and deterministic nonce into a reproducible 64-bit seed."""
    ...

def uniform(min: float, max: float) -> float:
    """Draw from a continuous uniform distribution on [min, max]."""
    ...

def triangular(min: float, mode: float, max: float) -> float:
    """Draw from a triangular distribution with endpoints and a peak mode."""
    ...

def normal(mu: float = 0.0, sigma: float = 1.0) -> float:
    """Draw from a normal distribution with mean mu and standard deviation sigma."""
    ...

def exponential(mean: float) -> float:
    """Draw from an exponential distribution with the given mean."""
    ...

def gamma(shape: float, scale: float = 1.0) -> float:
    """Draw from a gamma distribution with shape and scale parameters."""
    ...

def beta(a: float, b: float, min: float = 0.0, max: float = 1.0) -> float:
    """Draw from a beta distribution scaled to [min, max]."""
    ...

def pert(min: float, mode: float, max: float) -> float:
    """Draw from the standard PERT empirical distribution."""
    ...

def pert_mod(min: float, mode: float, max: float, lambda_: float) -> float:
    """Draw from a modified PERT distribution with an explicit lambda shape."""
    ...

def dice(min: int, max: int) -> int:
    """Draw an integer uniformly from the inclusive interval [min, max]."""
    ...

def flip() -> bool:
    """Draw an unbiased Bernoulli trial."""
    ...

def bernoulli(p: float) -> bool:
    """Draw a Bernoulli trial that is true with probability p."""
    ...

def time() -> float:
    """Return the current simulation clock time for the active event queue."""
    ...

def hold(duration: float) -> _ProcessSignal:
    """Suspend the current process for duration simulated time units."""
    ...

def yield_process() -> _ProcessSignal:
    """Yield the current process until another process, timer, or event resumes it."""
    ...

def process_exit(value: object | None = None) -> NoReturn:
    """Exit the current process immediately with an optional Python exit value."""
    ...

def current_process() -> Process | None:
    """Return the currently running process, or None outside process execution."""
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

    def stop_at(self, when: float, priority: _Priority = 0) -> _EventHandle:
        """Schedule a stop event at absolute simulation time when."""
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
        """Stop a running process and return the Cimba process signal."""
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

class Condition:
    """Condition variable for arbitrary Python demand predicates."""

    def __init__(self, name: str) -> None:
        """Create a named condition variable."""
        ...

    def wait(
        self,
        predicate: _ConditionPredicate,
        context: object | None = None,
    ) -> _ProcessSignal:
        """Wait until predicate(process, context) returns true.

        The predicate is re-evaluated when the condition is signaled. Processes
        should recheck state after wakeup because another process may have acted
        first.
        """
        ...

    def signal(self) -> _Count:
        """Evaluate waiting predicates and reactivate those that are true."""
        ...

    def close(self) -> None:
        """Destroy the native condition variable."""
        ...

class DataSummary:
    """Single-pass unweighted summary of sample moments."""

    def __init__(self) -> None:
        """Create an empty data summary."""
        ...

    def add(self, value: float) -> _Count:
        """Add one sample value and return the new sample count."""
        ...

    @property
    def count(self) -> _Count:
        """Number of samples summarized."""
        ...

    @property
    def min(self) -> float:
        """Smallest sample value seen."""
        ...

    @property
    def max(self) -> float:
        """Largest sample value seen."""
        ...

    @property
    def mean(self) -> float:
        """Sample mean."""
        ...

    @property
    def variance(self) -> float:
        """Sample variance."""
        ...

    @property
    def stddev(self) -> float:
        """Sample standard deviation."""
        ...

    @property
    def skewness(self) -> float:
        """Sample skewness."""
        ...

    @property
    def kurtosis(self) -> float:
        """Sample excess kurtosis."""
        ...

    def close(self) -> None:
        """Destroy the native data summary."""
        ...

class WeightedSummary:
    """Single-pass duration/weight-aware summary of sample moments."""

    def __init__(self) -> None:
        """Create an empty weighted summary."""
        ...

    def add(self, value: float, weight: float = 1.0) -> _Count:
        """Add one weighted sample and return the new sample count."""
        ...

    @property
    def count(self) -> _Count:
        """Number of weighted samples summarized."""
        ...

    @property
    def weight_sum(self) -> float:
        """Total accumulated weight."""
        ...

    @property
    def min(self) -> float:
        """Smallest sample value seen."""
        ...

    @property
    def max(self) -> float:
        """Largest sample value seen."""
        ...

    @property
    def mean(self) -> float:
        """Weighted mean."""
        ...

    @property
    def variance(self) -> float:
        """Weighted variance."""
        ...

    @property
    def stddev(self) -> float:
        """Weighted standard deviation."""
        ...

    @property
    def skewness(self) -> float:
        """Weighted skewness."""
        ...

    @property
    def kurtosis(self) -> float:
        """Weighted excess kurtosis."""
        ...

    def close(self) -> None:
        """Destroy the native weighted summary."""
        ...

class Dataset:
    """Resizable collection of unweighted float samples."""

    def __init__(self) -> None:
        """Create an empty dataset."""
        ...

    def add(self, value: float) -> _Count:
        """Add one sample and return the new sample count."""
        ...

    def values(self) -> list[float]:
        """Return the dataset's sample values in stored order."""
        ...

    def summary(self) -> DataSummary:
        """Compute and return an unweighted DataSummary for the dataset."""
        ...

    @property
    def count(self) -> _Count:
        """Number of stored samples."""
        ...

    @property
    def min(self) -> float:
        """Smallest stored sample value."""
        ...

    @property
    def max(self) -> float:
        """Largest stored sample value."""
        ...

    @property
    def median(self) -> float:
        """Median of the stored samples."""
        ...

    def close(self) -> None:
        """Destroy the native dataset."""
        ...

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
