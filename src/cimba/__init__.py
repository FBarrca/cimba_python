"""Python bindings for Cimba, a discrete-event-simulation library."""

Use :mod:`cimba.sim` for the SimPy-flavored modeling API.
"""

import os

from ._cimba import ffi, lib

__all__ = [
    "LOGGER_FATAL",
    "LOGGER_ERROR",
    "LOGGER_WARNING",
    "LOGGER_INFO",
    "logger_flags_on",
    "logger_flags_off",
    "native_version",
    "version",
    "use_threads",
    "__version__",
]

#: Version of this Python wrapper (distinct from the native Cimba version).
__version__ = "0.1.0"

LOGGER_FATAL = 0x80000000
LOGGER_ERROR = 0x40000000
LOGGER_WARNING = 0x20000000
LOGGER_INFO = 0x10000000


def version() -> str:
    """Return the cimba library version string."""
    return ffi.string(lib.cimba_version()).decode()


def native_version() -> str:
    """Return the version of the underlying Cimba C library."""
    return version()


def logger_flags_on(flags: int) -> None:
    """Turn on native logger flags for this thread and future trial threads."""
    lib.cpy_logger_flags_on(flags)


def logger_flags_off(flags: int) -> None:
    """Turn off native logger flags for this thread and future trial threads."""
    lib.cpy_logger_flags_off(flags)


def use_threads(n: int) -> int:
    """Return the number of worker threads Cimba will use.

    The upstream library always runs one worker thread per logical CPU core.
    The ``n`` argument is accepted for API compatibility (``0`` means all cores)
    but is not passed through to the C library yet.
    """
    return os.cpu_count() or 1
