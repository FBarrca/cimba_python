"""Exercise the actual exported input runtime through its C ABI."""

import ctypes

import numpy as np

from cimba.engine.abi import ABI_VERSION, INPUT_SLOT
from cimba.engine.library import load


def _library():
    library = load()
    library.cpy_abi_version.restype = ctypes.c_uint32
    library.cpy_input_slot_sizeof.restype = ctypes.c_size_t
    library.cpy_input_seed.argtypes = (ctypes.c_void_p, ctypes.c_uint64)
    library.cpy_input_next.argtypes = (ctypes.c_void_p,)
    library.cpy_input_next.restype = ctypes.c_double
    library.cpy_series_at.argtypes = (ctypes.c_void_p, ctypes.c_double)
    library.cpy_series_at.restype = ctypes.c_double
    return library


def test_abi_and_stream_independence():
    library = _library()
    assert library.cpy_abi_version() == ABI_VERSION
    assert library.cpy_input_slot_sizeof() == INPUT_SLOT.itemsize
    a = np.zeros(1, dtype=INPUT_SLOT)
    b = np.zeros(1, dtype=INPUT_SLOT)
    for slot in (a, b):
        slot["kind"] = 1
        slot["distribution"] = 1  # exponential
        slot["parameters"][0, 0] = 2.0
        library.cpy_input_seed(slot.ctypes.data, 301)
    first = library.cpy_input_next(a.ctypes.data)
    for _ in range(20):
        library.cpy_input_next(b.ctypes.data)
    second = library.cpy_input_next(a.ctypes.data)
    again = np.zeros(1, dtype=INPUT_SLOT)
    again["kind"] = 1
    again["distribution"] = 1
    again["parameters"][0, 0] = 2.0
    library.cpy_input_seed(again.ctypes.data, 301)
    assert first == library.cpy_input_next(again.ctypes.data)
    assert second == library.cpy_input_next(again.ctypes.data)


def test_rows_series_and_exhaustion():
    library = _library()
    row = np.asarray([4.0, 5.0, 6.0], dtype=np.float64)
    slot = np.zeros(1, dtype=INPUT_SLOT)
    slot["kind"] = 2
    slot["policy"] = 4  # extend
    slot["data"] = row.ctypes.data
    slot["length"] = len(row)
    slot["step"] = 1.0
    slot["last_bucket"] = -1
    assert library.cpy_series_at(slot.ctypes.data, 2.2) == 6.0
    assert slot["cursor"][0] == 3
    assert np.isnan(library.cpy_series_at(slot.ctypes.data, 3.0))
    assert slot["status"][0] == 3  # rerun after extension
    slot["status"] = 0
    slot["policy"] = 2  # wrap
    assert library.cpy_series_at(slot.ctypes.data, 3.0) == 4.0


def test_native_distributions_have_expected_means_and_isolated_streams():
    library = _library()
    cases = [
        (1, (2.0,), 2.0),          # exponential
        (2, (4.0, 1.5), 4.0),      # normal
        (3, (2.0, 3.0), 6.0),      # gamma
        (4, (0.0, 0.5), np.exp(0.125)),  # lognormal
        (5, (2.0, 1.0), np.sqrt(np.pi) / 2),  # Weibull
        (6, (40.0,), 40.0),       # Poisson, large-rate rejection path
        (7, (0.0, 1.0, 3.0), 4.0 / 3.0),
        (8, (0.0, 1.0, 2.0), 1.0),
    ]
    for distribution, parameters, expected in cases:
        slot = np.zeros(1, dtype=INPUT_SLOT)
        slot["kind"] = 1
        slot["distribution"] = distribution
        slot["parameters"][0, :len(parameters)] = parameters
        library.cpy_input_seed(slot.ctypes.data, 900 + distribution)
        draws = np.asarray([library.cpy_input_next(slot.ctypes.data)
                            for _ in range(20_000)])
        assert np.isfinite(draws).all()
        assert abs(draws.mean() - expected) < max(0.1, expected * 0.04)

    library.cpy_random01.restype = ctypes.c_double
    library.cmb_random_initialize.argtypes = (ctypes.c_uint64,)
    library.cmb_random_initialize(77)
    first = library.cpy_random01()
    slot = np.zeros(1, dtype=INPUT_SLOT)
    slot["kind"] = 1
    slot["distribution"] = 1
    slot["parameters"][0, 0] = 2
    library.cpy_input_seed(slot.ctypes.data, 432)
    library.cpy_input_next(slot.ctypes.data)
    second = library.cpy_random01()
    library.cmb_random_initialize(77)
    assert first == library.cpy_random01()
    assert second == library.cpy_random01()
