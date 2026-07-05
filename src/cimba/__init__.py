import os

from .cimba import (
    gil_enabled,
    native_version,
    run_experiment,
    run_native_experiment,
    set_native_thread_hooks,
)
from .cmb_buffer import Buffer, UNLIMITED
from .cmb_condition import Condition
from .cmb_datasummary import DataSummary
from .cmb_dataset import Dataset
from .cmb_event import Simulation, time
from .cmb_logger import (
    LOGGER_ERROR,
    LOGGER_FATAL,
    LOGGER_INFO,
    LOGGER_WARNING,
    logger_flags_off,
    logger_flags_on,
)
from .cmb_objectqueue import ObjectQueue
from .cmb_priorityqueue import PriorityQueue
from .cmb_process import (
    CANCELLED,
    INTERRUPTED,
    PREEMPTED,
    PROCESS_CREATED,
    PROCESS_FINISHED,
    PROCESS_RUNNING,
    STOPPED,
    SUCCESS,
    TIMEOUT,
    Process,
    current_process,
    hold,
    process_exit,
    wait_event,
    yield_process,
)
from .cmb_resource import Resource
from .cmb_resourcepool import ResourcePool
from .cmb_timeseries import TimeSeries
from .cmb_wtdsummary import WeightedSummary
from . import random as random
from . import reporting as reporting

__all__ = [
    "Buffer",
    "CANCELLED",
    "Condition",
    "DataSummary",
    "Dataset",
    "INTERRUPTED",
    "LOGGER_ERROR",
    "LOGGER_FATAL",
    "LOGGER_INFO",
    "LOGGER_WARNING",
    "ObjectQueue",
    "PREEMPTED",
    "PROCESS_CREATED",
    "PROCESS_FINISHED",
    "PROCESS_RUNNING",
    "PriorityQueue",
    "Process",
    "Resource",
    "ResourcePool",
    "STOPPED",
    "SUCCESS",
    "Simulation",
    "TIMEOUT",
    "TimeSeries",
    "UNLIMITED",
    "WeightedSummary",
    "current_process",
    "gil_enabled",
    "hold",
    "logger_flags_off",
    "logger_flags_on",
    "native_version",
    "process_exit",
    "random",
    "reporting",
    "run_experiment",
    "run_native_experiment",
    "set_native_thread_hooks",
    "time",
    "native_version",
    "use_threads",
    "version",
    "wait_event",
    "yield_process",
    "__version__",
]

#: Version of this Python wrapper (distinct from the native Cimba version).
__version__ = "0.3.0"


def version() -> str:
    """Return the cimba library version string."""
    return native_version()


def use_threads(n: int) -> int:
    """Return the number of worker threads Cimba will use.

    The upstream library always runs one worker thread per logical CPU core.
    The ``n`` argument is accepted for API compatibility (``0`` means all cores)
    but is not passed through to the C library yet.
    """
    return os.cpu_count() or 1
