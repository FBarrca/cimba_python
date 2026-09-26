"""Load the native library and expose its host ABI through ctypes."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from importlib import resources

from cimba.engine.abi import ABI_VERSION, INPUT_SLOT


@lru_cache(maxsize=1)
def load():
    directory = resources.files("cimba")
    candidates = ("libcimba_py.so", "libcimba_py.dylib", "cimba_py.dll")
    path = next((directory / name for name in candidates
                 if (directory / name).is_file()), None)
    if path is None:
        raise ImportError(f"standalone Cimba runtime missing from {directory}")
    library = ctypes.CDLL(str(path))
    library.cpy_abi_version.restype = ctypes.c_uint32
    library.cpy_input_slot_sizeof.restype = ctypes.c_size_t
    if library.cpy_abi_version() != ABI_VERSION:
        raise ImportError("Cimba native ABI version mismatch")
    if library.cpy_input_slot_sizeof() != INPUT_SLOT.itemsize:
        raise ImportError("Cimba InputSlot size mismatch")
    return library
