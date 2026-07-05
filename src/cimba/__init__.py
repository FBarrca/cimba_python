import os

from .cimba import (
    gil_enabled,
    native_version,
    run_experiment,
    run_native_experiment,
    set_native_thread_hooks,
)
from .cmb_logger import (
    LOGGER_ERROR,
    LOGGER_FATAL,
    LOGGER_INFO,
    LOGGER_WARNING,
    logger_flags_off,
    logger_flags_on,
)
from . import random as random

__all__ = [
    "LOGGER_ERROR",
    "LOGGER_FATAL",
    "LOGGER_INFO",
    "LOGGER_WARNING",
    "gil_enabled",
    "logger_flags_off",
    "logger_flags_on",
    "native_version",
    "random",
    "run_experiment",
    "run_native_experiment",
    "set_native_thread_hooks",
    "use_threads",
    "version",
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
