"""Runtime helpers for collect-declared native capture arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._cimba import ffi, lib

HISTORY_CAPTURE_TRIAL_FIELD = "_cimba_trial_index"
HISTORY_CAPTURE_STORE_FIELD = "_cimba_history_capture_store"


@dataclass(frozen=True)
class HistoryCaptureSpec:
    name: str
    binding: str
    slot: int
    columns: int = 3


def create_capture_store(num_trials: int, num_slots: int) -> Any:
    store = lib.cpy_history_capture_store_create(num_trials, num_slots)
    if store == ffi.NULL:
        raise MemoryError("could not allocate capture store")
    return store


def destroy_capture_store(store: Any) -> None:
    lib.cpy_history_capture_store_destroy(store)


def copy_capture_store(
    store: Any,
    *,
    num_trials: int,
    specs: tuple[HistoryCaptureSpec, ...],
) -> dict[str, tuple[np.ndarray, ...]]:
    captured: dict[str, tuple[np.ndarray, ...]] = {}
    itemsize = np.dtype(np.float64).itemsize
    for spec in specs:
        rows: list[np.ndarray] = []
        for trial in range(num_trials):
            count = int(lib.cpy_history_capture_store_count(
                store, trial, spec.slot))
            data = lib.cpy_history_capture_store_data(store, trial, spec.slot)
            if count == 0:
                shape = (0,) if spec.columns == 1 else (0, spec.columns)
                rows.append(np.empty(shape, dtype=np.float64))
                continue
            if data == ffi.NULL:
                raise MemoryError("capture data is missing")
            view = ffi.buffer(data, count * spec.columns * itemsize)
            arr = np.frombuffer(view, dtype=np.float64).copy()
            if spec.columns != 1:
                arr = arr.reshape(count, spec.columns)
            rows.append(arr)
        captured[spec.name] = tuple(rows)
    return captured
