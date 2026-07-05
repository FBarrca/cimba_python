"""Runtime helpers for captured native history arrays."""

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


def create_capture_store(num_trials: int, num_slots: int) -> Any:
    store = lib.cpy_history_capture_store_create(num_trials, num_slots)
    if store == ffi.NULL:
        raise MemoryError("could not allocate history capture store")
    return store


def destroy_capture_store(store: Any) -> None:
    lib.cpy_history_capture_store_destroy(store)


def copy_capture_store(
    store: Any,
    *,
    num_trials: int,
    names: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, ...]]:
    captured: dict[str, tuple[np.ndarray, ...]] = {}
    itemsize = np.dtype(np.float64).itemsize
    for slot, name in enumerate(names):
        rows: list[np.ndarray] = []
        for trial in range(num_trials):
            count = int(lib.cpy_history_capture_store_count(
                store, trial, slot))
            data = lib.cpy_history_capture_store_data(store, trial, slot)
            if count == 0:
                rows.append(np.empty((0, 3), dtype=np.float64))
                continue
            if data == ffi.NULL:
                raise MemoryError("history capture data is missing")
            view = ffi.buffer(data, count * 3 * itemsize)
            arr = np.frombuffer(view, dtype=np.float64).copy()
            rows.append(arr.reshape(count, 3))
        captured[name] = tuple(rows)
    return captured
