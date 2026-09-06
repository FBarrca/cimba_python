"""Checks for the native ABI consumed by compiled models and cached callbacks."""

import ctypes

from numba import types

import cimba
from cimba import _bindings, _cimba_native, _model


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


def test_callback_cache_changes_with_native_version(monkeypatch):
    from cimba import _cimba

    _model._callback_cache_platform_key.cache_clear()
    try:
        original = _model._callback_cache_platform_key()
        monkeypatch.setattr(_cimba, "native_version", lambda: "different-native-abi")
        _model._callback_cache_platform_key.cache_clear()
        assert _model._callback_cache_platform_key() != original
    finally:
        _model._callback_cache_platform_key.cache_clear()
