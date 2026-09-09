"""Checks for the native ABI consumed by compiled models."""

import ctypes

from numba import types

import cimba
from cimba import _bindings, _cimba_native


def test_native_runtime_is_rc2():
    assert cimba.native_version() == "3.0.0-RC2"


def test_all_numba_bindings_resolve_in_native_library():
    library = ctypes.CDLL(_cimba_native.__file__)
    missing = sorted({
        binding.symbol
        for binding in vars(_bindings).values()
        if isinstance(binding, types.ExternalFunction)
        and not hasattr(library, binding.symbol)
    })
    assert missing == []
