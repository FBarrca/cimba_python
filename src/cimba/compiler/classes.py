"""Compile one model class to native callbacks without inspecting its instances."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import inspect
from typing import Any, cast as type_cast

import numpy as np

from llvmlite import ir
from numba import carray, cfunc, from_dtype, njit, types
from numba.extending import intrinsic

from cimba.layout import RecordLayout
from cimba.engine.abi import HOOK_DESCRIPTOR, INPUT_DESCRIPTOR, PROCESS_DESCRIPTOR
from cimba.schema import ClassSchema
from . import handles  # register record handle operations
from . import spawning  # register dynamic Model class typing and spawn lowering
from .views import register_views
from .functions import compile_function


class ModelCompileError(RuntimeError):
    pass


def _cast_pointer(record_type):
    pointer_type = types.CPointer(record_type)

    @intrinsic
    def cast(typing_context, address):
        if not isinstance(address, types.Integer):
            raise TypeError("model view address must be an integer")

        def codegen(context, builder, signature, arguments):
            return builder.inttoptr(arguments[0], context.get_value_type(pointer_type))

        return pointer_type(address), codegen

    return cast


@dataclass(frozen=True)
class CompiledClass:
    schema: ClassSchema
    layout: RecordLayout
    processes: dict[str, Any]
    starts: dict[str, Any]
    ends: dict[str, Any]
    events: dict[str, Any]
    predicates: dict[str, Any]
    dispatch: np.ndarray
    functions: dict[str, Any]
    dynamic_processes: np.ndarray
    dynamic_starts: np.ndarray
    dynamic_ends: np.ndarray
    dynamic_inputs: np.ndarray


@lru_cache(maxsize=None)
def ensure(schema: ClassSchema) -> CompiledClass:
    """The cache key contains class declarations only, never an Assembly.

    Native descriptors retain callback addresses. Keep callbacks pinned until
    an explicit cache clear so compiling many classes cannot evict live code.
    """
    layout = RecordLayout.of(schema)
    register_views(schema)
    record_type = from_dtype(layout.dtype)
    cast = type_cast(Any, _cast_pointer(record_type))
    processes = {}
    starts = {}
    ends = {}
    events = {}
    predicates = {}
    functions = {}
    dispatch = np.zeros(max(1, len(schema.slots)), dtype=np.uintp)
    for name in schema.events:
        index = schema.slots.index(name)
        method = getattr(schema.cls, name)
        try:
            compiled = type_cast(Any, njit(types.void(record_type))(method))

            @cfunc(types.void(types.intp), no_cpython_wrapper=True)
            def callback(context):
                view = carray(cast(context), 1)[0]
                compiled(view)

            events[name] = callback
            dispatch[index] = type_cast(Any, callback).address
        except Exception as exc:
            raise _compile_error(schema.cls, method, exc) from exc
    for name in schema.predicates:
        index = schema.slots.index(name)
        method = getattr(schema.cls, name)
        try:
            compiled = type_cast(Any, njit(types.boolean(record_type))(method))

            @cfunc(types.uint8(types.intp), no_cpython_wrapper=True)
            def callback(context):
                view = carray(cast(context), 1)[0]
                return 1 if compiled(view) else 0

            predicates[name] = callback
            dispatch[index] = type_cast(Any, callback).address
        except Exception as exc:
            raise _compile_error(schema.cls, method, exc) from exc
    for name in schema.functions:
        method = getattr(schema.cls, name)
        try:
            functions[name] = compile_function(schema, name, record_type, cast)
            dispatch[schema.slots.index(name)] = type_cast(Any, functions[name]).address
        except Exception as exc:
            raise _compile_error(schema.cls, method, exc) from exc
    for name, _, _ in schema.processes:
        method = getattr(schema.cls, name)
        try:
            compiled = type_cast(Any, njit(types.void(record_type))(method))

            @cfunc(types.intp(types.intp, types.intp),
                   no_cpython_wrapper=True)
            def callback(process, context):
                view = carray(cast(context), 1)[0]
                compiled(view)
                return 0

            processes[name] = callback
        except Exception as exc:
            raise _compile_error(schema.cls, method, exc) from exc
    for names, destination in ((schema.starts, starts), (schema.ends, ends)):
        for name in names:
            method = getattr(schema.cls, name)
            try:
                compiled = type_cast(Any, njit(types.void(record_type))(method))

                @cfunc(types.void(types.intp), no_cpython_wrapper=True)
                def callback(context):
                    view = carray(cast(context), 1)[0]
                    compiled(view)

                destination[name] = callback
            except Exception as exc:
                raise _compile_error(schema.cls, method, exc) from exc
    dispatch.flags.writeable = False
    dynamic_processes = np.array([
        (0, processes[name].address, priority, copies,
         name.encode("utf-8")[:31])
        for name, copies, priority in schema.processes], dtype=PROCESS_DESCRIPTOR)
    dynamic_starts = np.array(
        [(0, starts[name].address) for name in schema.starts],
        dtype=HOOK_DESCRIPTOR)
    dynamic_ends = np.array(
        [(0, ends[name].address) for name in schema.ends],
        dtype=HOOK_DESCRIPTOR)
    dynamic_inputs = np.array(
        [(0, layout.offset(field.name),
          f"{schema.cls.__name__}.{field.name}".encode("utf-8")[:95])
         for field in schema.fields if field.kind in {"input", "series"}],
        dtype=INPUT_DESCRIPTOR)
    for table in (dynamic_processes, dynamic_starts, dynamic_ends,
                  dynamic_inputs):
        table.flags.writeable = False
    return CompiledClass(schema, layout, processes, starts, ends,
                         events, predicates, dispatch, functions, dynamic_processes,
                         dynamic_starts, dynamic_ends, dynamic_inputs)


def _compile_error(cls, method, original: Exception) -> ModelCompileError:
    filename = inspect.getsourcefile(method) or "<unknown>"
    try:
        line = inspect.getsourcelines(method)[1]
    except (OSError, TypeError):
        line = 0
    return ModelCompileError(
        f"{filename}:{line}: {cls.__name__}.{method.__name__}: {original}")
