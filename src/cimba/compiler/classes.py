"""Compile one model class to native entry points without inspecting its instances."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import inspect
from types import MappingProxyType

import numpy as np

from numba import from_dtype, njit, types

from cimba.layout import RecordLayout
from cimba.engine.abi import HOOK_DESCRIPTOR, INPUT_DESCRIPTOR, PROCESS_DESCRIPTOR
from cimba.schema import ClassSchema
from . import handles  # register record handle operations
from . import spawning  # register dynamic Model class typing and spawn lowering
from .entries import EntryModule
from .views import register_reachable_views
from .functions import function_signature


class ModelCompileError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompiledClass:
    schema: ClassSchema
    layout: RecordLayout
    entries: Mapping[str, int]
    dispatch: np.ndarray
    dynamic_processes: np.ndarray
    dynamic_starts: np.ndarray
    dynamic_ends: np.ndarray
    dynamic_inputs: np.ndarray
    native: EntryModule


def _signatures(schema: ClassSchema, record_type):
    """Yield ``(name, njit signature, is_process)`` for every compiled method."""
    for name in schema.events:
        yield name, types.void(record_type), False
    for name in schema.predicates:
        yield name, types.boolean(record_type), False
    for name in schema.functions:
        yield name, function_signature(schema, name, record_type), False
    for name, _, _ in schema.processes:
        yield name, types.void(record_type), True
    for name in schema.starts + schema.ends:
        yield name, types.void(record_type), False


@lru_cache(maxsize=None)
def ensure(schema: ClassSchema) -> CompiledClass:
    """The cache key contains class declarations only, never an Assembly.

    Each method is compiled once with ``njit``; its native entry point is a
    thin adapter emitted with the rest of the class (see ``entries``). Native
    descriptors retain entry addresses, so compiled classes stay pinned until
    an explicit cache clear.
    """
    layout = RecordLayout.of(schema)
    register_reachable_views(schema)
    record_type = from_dtype(layout.dtype)
    native = EntryModule(schema.cls.__qualname__)
    for name, signature, is_process in _signatures(schema, record_type):
        method = getattr(schema.cls, name)
        try:
            native.add(name, njit(signature, no_cpython_wrapper=True)(method),
                       process=is_process)
        except Exception as exc:
            raise _compile_error(schema.cls, method, exc) from exc
    entries = MappingProxyType(native.finalize())
    dispatch = np.zeros(max(1, len(schema.slots)), dtype=np.uintp)
    for name in schema.events + schema.predicates + schema.functions:
        dispatch[schema.slots.index(name)] = entries[name]
    dynamic_processes = np.array([
        (0, entries[name], priority, copies, name.encode("utf-8")[:31])
        for name, copies, priority in schema.processes], dtype=PROCESS_DESCRIPTOR)
    dynamic_starts = np.array(
        [(0, entries[name]) for name in schema.starts], dtype=HOOK_DESCRIPTOR)
    dynamic_ends = np.array(
        [(0, entries[name]) for name in schema.ends], dtype=HOOK_DESCRIPTOR)
    dynamic_inputs = np.array(
        [(0, layout.offset(field.name),
          f"{schema.cls.__name__}.{field.name}".encode("utf-8")[:95])
         for field in schema.fields if field.kind in {"input", "series"}],
        dtype=INPUT_DESCRIPTOR)
    for table in (dispatch, dynamic_processes, dynamic_starts, dynamic_ends,
                  dynamic_inputs):
        table.flags.writeable = False
    return CompiledClass(schema, layout, entries, dispatch, dynamic_processes,
                         dynamic_starts, dynamic_ends, dynamic_inputs, native)


def _compile_error(cls, method, original: Exception) -> ModelCompileError:
    filename = inspect.getsourcefile(method) or "<unknown>"
    try:
        line = inspect.getsourcelines(method)[1]
    except (OSError, TypeError):
        line = 0
    return ModelCompileError(
        f"{filename}:{line}: {cls.__name__}.{method.__name__}: {original}")
