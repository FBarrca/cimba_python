"""Cimba model runtime configuration and compiled random draws."""

from operator import index as _index

from ._cimba import ffi as _ffi, lib as _lib
from . import random as random

LOGGER_FATAL = 0x80000000
LOGGER_ERROR = 0x40000000
LOGGER_WARNING = 0x20000000
LOGGER_INFO = 0x10000000

__all__ = [
    "LOGGER_ERROR",
    "LOGGER_FATAL",
    "LOGGER_INFO",
    "LOGGER_WARNING",
    "logger_flags_off",
    "logger_flags_on",
    "native_version",
    "random",
    "use_threads",
    "version",
    "__version__",
]

#: Version of this Python wrapper (distinct from the native Cimba version).
__version__ = "0.5.10"


def native_version() -> str:
    """Return the bundled Cimba engine version."""
    return _ffi.string(_lib.cimba_version()).decode("utf-8")


def logger_flags_on(flags: int) -> None:
    """Enable logging categories for subsequent model runs."""
    _lib.cpy_logger_flags_on(_index(flags))


def logger_flags_off(flags: int) -> None:
    """Disable logging categories for subsequent model runs."""
    _lib.cpy_logger_flags_off(_index(flags))


def version() -> str:
    """Return the cimba library version string."""
    return native_version()


def use_threads(n: int) -> int:
    """Set native workers for the next run and return the effective count.

    Call between runs. Zero selects the runtime's CPU-count default.
    """
    n = _index(n)
    if not 0 <= n <= 0xFFFFFFFF:
        raise ValueError("worker count must fit an unsigned 32-bit integer")
    return int(_lib.cimba_threads_use(n))
