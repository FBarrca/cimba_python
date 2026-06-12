"""Python bindings for Cimba, a multithreaded discrete-event-simulation library.

Use :mod:`cimba.sim` for the SimPy-flavored modeling API.
"""

import os

from ._cimba import ffi, lib

__all__ = [
    "native_version",
    "version",
    "use_threads",
    "__version__",
]

#: Version of this Python wrapper (distinct from the native Cimba version).
__version__ = "0.1.0"


def version() -> str:
    """Return the cimba library version string."""
    return ffi.string(lib.cimba_version()).decode()


def native_version() -> str:
    """Return the version of the underlying Cimba C library."""
    return version()


def use_threads(n: int) -> int:
    """Return the number of worker threads Cimba will use.

    The upstream library always runs one worker thread per logical CPU core.
    The ``n`` argument is accepted for API compatibility (``0`` means all cores)
    but is not passed through to the C library yet.
    """
    return os.cpu_count() or 1
