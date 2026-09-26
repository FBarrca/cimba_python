"""The only Python-to-C host call site for trial execution."""

from __future__ import annotations

import ctypes

import numpy as np

from .abi import CAPTURE_SERIES, INPUT_SLOT
from .library import load


def run_blocks(blocks: np.ndarray, *, workers: int | None = None) -> int:
    if not blocks.flags.c_contiguous:
        raise ValueError("trial blocks must be contiguous")
    if workers is not None and workers < 1:
        raise ValueError("workers must be positive")
    library = load()
    library.cpy_run.argtypes = (ctypes.c_void_p, ctypes.c_uint64,
                                ctypes.c_size_t, ctypes.c_uint32)
    library.cpy_run.restype = ctypes.c_uint64
    return int(library.cpy_run(blocks.ctypes.data, len(blocks),
                               blocks.dtype.itemsize, workers or 0))


def seed_input(address: int, seed: int) -> None:
    library = load()
    library.cpy_input_seed.argtypes = (ctypes.c_void_p, ctypes.c_uint64)
    library.cpy_input_seed(address, seed)


def read_captures(headers: np.ndarray, entity_count: int):
    """Detach captured native arrays before their owning trial blocks leave scope."""
    library = load()
    library.cpy_capture_release.argtypes = (ctypes.c_void_p, ctypes.c_uint64)
    library.cpy_capture_release.restype = None
    captures = []
    for header in headers:
        address = int(header["capture"])
        if not address:
            captures.append(None)
            continue
        try:
            native = (ctypes.c_uint8 * (entity_count * CAPTURE_SERIES.itemsize))
            descriptors = np.ndarray((entity_count,), dtype=CAPTURE_SERIES,
                                     buffer=native.from_address(address))
            trial = []
            for descriptor in descriptors:
                count = int(descriptor["count"])
                def copied(pointer):
                    if not pointer or not count:
                        return np.empty(0, dtype=np.float64)
                    return np.ctypeslib.as_array(
                        (ctypes.c_double * count).from_address(int(pointer))).copy()
                trial.append((copied(descriptor["time"]),
                              copied(descriptor["value"])))
            captures.append(tuple(trial))
        finally:
            library.cpy_capture_release(address, entity_count)
            header["capture"] = 0
    return tuple(captures)


def distribution_rows(distribution: int, parameters: tuple[float, ...],
                      seed: int, length: int,
                      categorical: tuple[tuple[float, ...], tuple[float, ...]] | None = None
                      ) -> np.ndarray:
    if length < 0:
        raise ValueError("length must be nonnegative")
    slot = np.zeros(1, dtype=INPUT_SLOT)
    slot["kind"] = 1
    slot["distribution"] = distribution
    keepalive = None
    if categorical is None:
        slot["parameters"][0, :len(parameters)] = parameters
    else:
        values, weights = categorical
        keepalive = np.concatenate((values, weights)).astype(np.float64)
        slot["data"] = keepalive.ctypes.data
        slot["length"] = len(values)
    seed_input(slot.ctypes.data, seed)
    library = load()
    library.cpy_input_next.argtypes = (ctypes.c_void_p,)
    library.cpy_input_next.restype = ctypes.c_double
    values = np.empty(length, dtype=np.float64)
    for i in range(length):
        values[i] = library.cpy_input_next(slot.ctypes.data)
    return values
