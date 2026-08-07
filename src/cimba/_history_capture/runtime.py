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
    #: ``None`` for a scalar capture; a one-dimensional shape for an
    #: indexed component collection.  Collection items occupy contiguous
    #: slots beginning at ``slot``.
    shape: tuple[int, ...] | None = None

    @property
    def slot_count(self) -> int:
        if self.shape is None:
            return 1
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count


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
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    itemsize = np.dtype(np.float64).itemsize

    def copy_slot(slot: int, columns: int) -> np.ndarray:
        count = int(lib.cpy_history_capture_store_count(
            store, trial, slot))
        data = lib.cpy_history_capture_store_data(store, trial, slot)
        if count == 0:
            shape = (0,) if columns == 1 else (0, columns)
            return np.empty(shape, dtype=np.float64)
        if data == ffi.NULL:
            raise MemoryError("capture data is missing")
        view = ffi.buffer(data, count * columns * itemsize)
        arr = np.frombuffer(view, dtype=np.float64).copy()
        if columns != 1:
            arr = arr.reshape(count, columns)
        return arr

    for spec in specs:
        rows: list[Any] = []
        for trial in range(num_trials):
            if spec.shape is None:
                rows.append(copy_slot(spec.slot, spec.columns))
            else:
                if len(spec.shape) != 1:
                    raise ValueError(
                        "indexed history captures must have one dimension")
                rows.append(tuple(
                    copy_slot(spec.slot + index, spec.columns)
                    for index in range(spec.shape[0])))
        captured[spec.name] = tuple(rows)
    return captured
