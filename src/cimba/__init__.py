from operator import index

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
__version__ = "0.5.10"


def version() -> str:
    """Return the cimba library version string."""
    return native_version()


def use_threads(n: int) -> int:
    """Set native workers for the next run and return the effective count.

    Call between runs. Zero selects the runtime's CPU-count default.
    """
    from ._cimba import lib

    n = index(n)
    if not 0 <= n <= 0xFFFFFFFF:
        raise ValueError("worker count must fit an unsigned 32-bit integer")
    return int(lib.cimba_threads_use(n))
