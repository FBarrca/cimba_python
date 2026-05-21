"""Type stubs for the compiled ``cimba._cimba`` Cython extension.

Editors can't read signatures or docstrings out of the compiled ``.so``, so this
stub provides them for hover/autocomplete. Keep it in sync with ``_cimba.pyx``.
"""

def native_version() -> str:
    """Return the version of the underlying Cimba C library (e.g. ``"3.0.0-beta"``).

    This is the version of the native library this wheel was built against, as
    reported by ``cimba_version()`` in C. It is distinct from
    ``cimba.__version__``, which is the version of this Python wrapper.
    """
    ...
