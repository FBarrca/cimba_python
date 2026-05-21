# cython: language_level=3
"""Cython extension for the Cimba wrapper.

Scaffolding phase: this module exists to prove the build pipeline end to end
(git submodule -> Meson subproject -> Cython -> linked .so). It exposes a
single call into the native library so a successful import demonstrates real
linkage, not just that Cython ran. Real bindings come in later phases.
"""


cdef extern from "cimba.h":
    const char *cimba_version()


def native_version() -> str:
    """Return the version of the underlying Cimba C library (e.g. ``"3.0.0-beta"``).

    This is the version of the native library this wheel was built against, as
    reported by ``cimba_version()`` in C. It is distinct from
    :data:`cimba.__version__`, which is the version of this Python wrapper.
    """
    cdef const char *v = cimba_version()
    return v.decode("utf-8")
