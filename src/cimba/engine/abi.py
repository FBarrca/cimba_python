"""Pure NumPy descriptions of the versioned native trial ABI."""

from __future__ import annotations

import numpy as np
from typing import Any

ABI_VERSION = 1
POINTER = np.uintp

INPUT_SLOT = np.dtype([
    ("kind", np.uint32), ("policy", np.uint32),
    ("distribution", np.uint32), ("status", np.uint32),
    ("parameters", np.float64, (4,)),
    ("stream_state", np.uint64, (4,)),
    ("data", POINTER), ("length", np.uint64), ("cursor", np.uint64),
    ("step", np.float64), ("origin", np.float64),
    ("last_bucket", np.int64), ("last_value", np.float64),
    ("history", POINTER), ("history_capacity", np.uint64),
], align=True)


def _typed_input_slot(tag: str) -> np.dtype:
    slot_fields = INPUT_SLOT.fields
    slot_names = INPUT_SLOT.names
    assert slot_fields is not None and slot_names is not None
    fields: list[tuple[Any, Any]] = [((f"__cimba_input_{tag}", "kind"), np.uint32)]
    fields.extend((name, slot_fields[name][0])  # pyright: ignore[reportArgumentType]
                  for name in slot_names if name != "kind")
    result = np.dtype(fields, align=True)
    assert result.itemsize == INPUT_SLOT.itemsize
    return result


INPUT_SLOT_INT = _typed_input_slot("int")
INPUT_SLOT_BOOL = _typed_input_slot("bool")

TRIAL_HEADER = np.dtype([
    ("seed", np.uint64), ("trial_index", np.uint64),
    ("warmup", np.float64), ("duration", np.float64),
    ("cooldown", np.float64), ("status", np.uint32),
    ("error", "S256"), ("capture", POINTER),
    ("descriptor", POINTER),
], align=True)

ENTITY_DESCRIPTOR = np.dtype([
    ("record_offset", np.uint64), ("field_offset", np.uint64),
    ("kind", np.uint32), ("reserved", np.uint32),
    ("capacity", np.uint64), ("name", "S32"),
], align=True)

CAPTURE_SERIES = np.dtype([
    ("count", np.uint64), ("time", POINTER), ("value", POINTER),
], align=True)

PROCESS_DESCRIPTOR = np.dtype([
    ("record_offset", np.uint64), ("callback", POINTER),
    ("priority", np.int64), ("copies", np.uint64), ("name", "S32"),
], align=True)

HOOK_DESCRIPTOR = np.dtype([
    ("record_offset", np.uint64), ("callback", POINTER),
], align=True)

INPUT_DESCRIPTOR = np.dtype([
    ("record_offset", np.uint64), ("field_offset", np.uint64),
    ("name", "S96"),
], align=True)

TRIAL_DESCRIPTOR = np.dtype([
    ("entities", POINTER), ("entity_count", np.uint64),
    ("processes", POINTER), ("process_count", np.uint64),
    ("starts", POINTER), ("start_count", np.uint64),
    ("ends", POINTER), ("end_count", np.uint64),
    ("inputs", POINTER), ("input_count", np.uint64),
], align=True)
