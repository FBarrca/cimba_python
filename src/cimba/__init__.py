"""Python bindings for Cimba, a multithreaded discrete-event-simulation library.

This is the scaffolding release: it wires up the build against the native C
library but does not yet wrap the simulation API. ``native_version()`` calls
into the linked C library and is here to confirm the toolchain works end to end.
"""

from ._cimba import native_version

__all__ = ["native_version", "__version__"]

#: Version of this Python wrapper (distinct from the native Cimba version,
#: which is reported by :func:`version`).
__version__ = "0.1.0"
