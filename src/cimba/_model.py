"""Model declaration, compilation, and experiment execution.

A ``Model`` collects declared entities, parameters, outputs, and process
functions, then compiles everything on first ``experiment()``:

* component process bodies and data-only lifecycle callbacks are planned from
  the first normally constructed model and compiled once per model class;
  instance-specific callbacks are compiled on the first experiment;
* a fixed lifecycle ABI consumes runtime descriptor tables to create entities,
  schedule the recording window, start processes, run the event queue, collect
  statistics, and tear everything down without per-layout source generation;
* an ``Experiment`` is a structured numpy array with one record per trial
  (the ``env`` seen by process bodies) handed to ``cimba_run_experiment``,
  which runs trials in parallel across all cores.
"""

import ast
import copy
import hashlib
import importlib.metadata
import inspect
import os
import pickle
import platform
import sys
import tempfile
import textwrap
import threading
import time
import types as pytypes
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (TYPE_CHECKING, Any, Generic, Self, TypedDict, TypeVar,
                    cast, get_type_hints, overload)

import llvmlite
import numba
import numpy as np
from numpy.typing import ArrayLike

from numba import carray, cfunc, from_dtype, njit, types
from numba.extending import overload as _nb_overload

from . import _bindings as _b
from ._cimba import ffi, lib
from ._components import (
    Component,
    _COLLECT_ATTR,
    _EVENT_ATTR,
    _PREDICATE_ATTR,
    _PROCESS_ATTR,
    _build_component_functions,
    _class_declarations,
    _ComponentDecl,
    _component_collect_methods,
    _component_process_methods,
    _closure_namespace,
    _compile_lowered,
    _function_def_from_source,
    _lower_component_collect,
    _lower_component_process,
    _lower_dataset_methods,
    _lower_entity_methods,
    _lower_history_methods,
    _lower_model_component_refs,
    _marked_methods,
)
from ._declarations import (
    Handle,
    _MISSING,
    _Capacities,
    _check_name,
    _FIELD_KINDS,
    _FieldDecl,
    _STANDARD_FIELDS,
    Spawnable,
    class_type_hints,
)
from ._graph import (ProcessDAG, ProcessDAGBlock, ProcessDAGEdge,
                     ProcessDAGNode, infer_process_dag)
from ._history_capture import (
    HISTORY_CAPTURE_STORE_FIELD,
    HISTORY_CAPTURE_TRIAL_FIELD,
    HistoryCaptureSpec,
    copy_capture_store,
    create_capture_store,
    destroy_capture_store,
    lower_dataset_capture_methods,
    lower_history_capture_methods,
)
from ._intrinsics import addressof, call_void_callback, ptr_caster
from .random._lowering import lower_random_calls_in_function

_F = TypeVar("_F", bound=Callable[..., Any])

_RECORDING_EVENT_FIELD = "_cimba_recording_event"
_TRIAL_INITIALIZE_FIELD = "_cimba_trial_initialize"
_TRIAL_ENTITIES_FIELD = "_cimba_trial_entities"
_TRIAL_PROCESSES_FIELD = "_cimba_trial_processes"
_TRIAL_TEARDOWN_FIELD = "_cimba_trial_teardown"
_TRIAL_STOP_FIELD = "_cimba_trial_stop"
_TRIAL_PROCESS_CLEANUP_FIELD = "_cimba_trial_process_cleanup"
_TRIAL_COLLECT_FIELD = "_cimba_trial_collect"
_PROCESS_DESCRIPTORS_FIELD = "_cimba_process_descriptors"
_PROCESS_DESCRIPTOR_COUNT_FIELD = "_cimba_process_descriptor_count"
_PROCESS_HANDLES_FIELD = "_cimba_process_handles"
_PROCESS_HANDLE_COUNT_FIELD = "_cimba_process_handle_count"
_PROCESS_CONTEXTS_FIELD = "_cimba_process_contexts"
_HAS_SPAWNED_FIELD = "_cimba_has_spawned"
_COLLECT_DESCRIPTORS_FIELD = "_cimba_collect_descriptors"
_COLLECT_DESCRIPTOR_COUNT_FIELD = "_cimba_collect_descriptor_count"
_ENTITY_DESCRIPTORS_FIELD = "_cimba_entity_descriptors"
_ENTITY_DESCRIPTOR_COUNT_FIELD = "_cimba_entity_descriptor_count"
_RUNTIME_TEXT_HANDLES_FIELD = "_cimba_runtime_text_handles"

_LIFECYCLE_ABI_FIELDS = [
    *_STANDARD_FIELDS,
    (_RECORDING_EVENT_FIELD, "<i8"),
    (_TRIAL_INITIALIZE_FIELD, "<i8"),
    (_TRIAL_ENTITIES_FIELD, "<i8"),
    (_TRIAL_PROCESSES_FIELD, "<i8"),
    (_TRIAL_TEARDOWN_FIELD, "<i8"),
    (_TRIAL_STOP_FIELD, "<i8"),
    (_TRIAL_PROCESS_CLEANUP_FIELD, "<i8"),
    (_TRIAL_COLLECT_FIELD, "<i8"),
    (_PROCESS_DESCRIPTORS_FIELD, "<i8"),
    (_PROCESS_DESCRIPTOR_COUNT_FIELD, "<i8"),
    (_PROCESS_HANDLES_FIELD, "<i8"),
    (_PROCESS_HANDLE_COUNT_FIELD, "<i8"),
    (_PROCESS_CONTEXTS_FIELD, "<i8"),
    (_HAS_SPAWNED_FIELD, "<i8"),
    (_COLLECT_DESCRIPTORS_FIELD, "<i8"),
    (_COLLECT_DESCRIPTOR_COUNT_FIELD, "<i8"),
    (_ENTITY_DESCRIPTORS_FIELD, "<i8"),
    (_ENTITY_DESCRIPTOR_COUNT_FIELD, "<i8"),
    (_RUNTIME_TEXT_HANDLES_FIELD, "<i8"),
]
_LIFECYCLE_ABI_DTYPE = np.dtype(_LIFECYCLE_ABI_FIELDS)
_LIFECYCLE_ABI_RECORD = from_dtype(_LIFECYCLE_ABI_DTYPE)
_LIFECYCLE_ABI_PTR = types.CPointer(_LIFECYCLE_ABI_RECORD)
_INT64_FROM_ADDRESS = ptr_caster(types.int64)
_FLOAT64_FROM_ADDRESS = ptr_caster(types.float64)


@njit(inline="always")
def _runtime_text_handle(table_address, slot):
    """Read a process-local text pointer from a model sidecar table."""
    return carray(_INT64_FROM_ADDRESS(table_address), slot + 1)[slot]


_PROCESS_DESCRIPTOR_WIDTH = 8
_PD_CALLBACK = 0
_PD_NAME = 1
_PD_ALLOC_SIZE = 2
_PD_PRIORITY = 3
_PD_COPIES = 4
_PD_INDEXED = 5
_PD_HANDLE_START = 6
_PD_DEST_OFFSET = 7

_ENTITY_DESCRIPTOR_WIDTH = 5
_ED_KIND = 0
_ED_NAME = 1
_ED_CAPACITY_MODE = 2
_ED_CAPACITY = 3
_ED_DEST_OFFSET = 4

_ENTITY_BUFFER = 1
_ENTITY_RESOURCE = 2
_ENTITY_RESOURCEPOOL = 3
_ENTITY_OBJECTQUEUE = 4
_ENTITY_DATASET = 5
_ENTITY_CONDITION = 6
_ENTITY_PRIORITYQUEUE = 7
_CAPACITY_CONSTANT = 0
_CAPACITY_FIELD = 1


# Offset of derived-struct fields inside an extended process allocation:
# the cmb_process header, rounded up to the 8-byte record alignment.
_PROC_DATA_OFFSET = (int(lib.cpy_process_sizeof()) + 7) & ~7

# Native ``cmb_process.name`` and ``cmi_resourcebase.name`` (every named
# entity: Queue, Store, Resource, Pool, Condition, PQueues) are both 32-byte
# buffers including the trailing NUL. Logical Python names stay intact for
# fields, graphs, and diagnostics; only the runtime display name is shortened
# when necessary. Passing an over-long name through instead trips
# cmb_assert_release in cmi_resourcebase_set_name, which aborts the process --
# no Python exception, no traceback.
_NATIVE_NAME_BYTES = 31


def _native_cfunc(signature):
    """Build a callback without an unused Python-callable wrapper."""
    return cfunc(signature, no_cpython_wrapper=True)


def _compile_parallel_cfunc(signature, function):
    """Compile one callback in either the parent or a forked worker."""
    return _native_cfunc(signature)(function)


def _runtime_trial_processes(vtrl):
    """Create every scheduled process through the fixed lifecycle ABI."""
    env = carray(vtrl, 1)[0]
    descriptor_count = env[_PROCESS_DESCRIPTOR_COUNT_FIELD]
    handle_count = env[_PROCESS_HANDLE_COUNT_FIELD]
    self_addr = addressof(vtrl)
    descriptors = carray(
        _INT64_FROM_ADDRESS(env[_PROCESS_DESCRIPTORS_FIELD]),
        max(1, descriptor_count * _PROCESS_DESCRIPTOR_WIDTH),
    )
    handles = carray(
        _INT64_FROM_ADDRESS(env[_PROCESS_HANDLES_FIELD]), max(1, handle_count))
    contexts = carray(
        _INT64_FROM_ADDRESS(env[_PROCESS_CONTEXTS_FIELD]),
        max(1, handle_count * 2),
    )
    for descriptor_index in range(descriptor_count):
        base = descriptor_index * _PROCESS_DESCRIPTOR_WIDTH
        copies = descriptors[base + _PD_COPIES]
        first = descriptors[base + _PD_HANDLE_START]
        for copy_index in range(copies):
            slot = first + copy_index
            if descriptors[base + _PD_INDEXED] != 0:
                contexts[2 * slot] = self_addr
                contexts[2 * slot + 1] = copy_index
                context = env[_PROCESS_CONTEXTS_FIELD] + 16 * slot
            else:
                context = self_addr
            alloc_size = descriptors[base + _PD_ALLOC_SIZE]
            if alloc_size == _PROC_DATA_OFFSET:
                process = _b.process_create()
            else:
                process = _b.process_create_sized(alloc_size)
            _b.process_initialize(
                process,
                descriptors[base + _PD_NAME],
                descriptors[base + _PD_CALLBACK],
                context,
                descriptors[base + _PD_PRIORITY],
            )
            _b.process_start(process)
            handles[slot] = process
            destination = descriptors[base + _PD_DEST_OFFSET]
            if destination >= 0:
                target = carray(
                    _INT64_FROM_ADDRESS(
                        self_addr + destination + 8 * copy_index),
                    1,
                )
                target[0] = process


def _runtime_trial_stop(vtrl):
    """Stop live scheduled and spawned processes through the stable ABI."""
    env = carray(vtrl, 1)[0]
    handle_count = env[_PROCESS_HANDLE_COUNT_FIELD]
    handles = carray(
        _INT64_FROM_ADDRESS(env[_PROCESS_HANDLES_FIELD]), max(1, handle_count))
    for index in range(handle_count):
        if _b.process_status(handles[index]) == 1:
            _b.process_stop(handles[index], 0)
    if env[_HAS_SPAWNED_FIELD] != 0:
        _b.spawned_stop_all()


def _runtime_trial_process_cleanup(vtrl):
    """Destroy scheduled processes and reclaim spawned processes."""
    env = carray(vtrl, 1)[0]
    handle_count = env[_PROCESS_HANDLE_COUNT_FIELD]
    handles = carray(
        _INT64_FROM_ADDRESS(env[_PROCESS_HANDLES_FIELD]), max(1, handle_count))
    for index in range(handle_count):
        _b.process_terminate(handles[index])
        _b.process_destroy(handles[index])
    if env[_HAS_SPAWNED_FIELD] != 0:
        _b.spawned_reclaim()


def _runtime_trial_collect(vtrl):
    """Invoke every end-of-trial collector through a fixed address table."""
    env = carray(vtrl, 1)[0]
    count = env[_COLLECT_DESCRIPTOR_COUNT_FIELD]
    callbacks = carray(
        _INT64_FROM_ADDRESS(env[_COLLECT_DESCRIPTORS_FIELD]), max(1, count))
    for index in range(count):
        call_void_callback(callbacks[index], vtrl)


def _runtime_trial_entities(vtrl):
    """Create model entities from layout-independent runtime descriptors."""
    env = carray(vtrl, 1)[0]
    self_addr = addressof(vtrl)
    count = env[_ENTITY_DESCRIPTOR_COUNT_FIELD]
    descriptors = carray(
        _INT64_FROM_ADDRESS(env[_ENTITY_DESCRIPTORS_FIELD]),
        max(1, count) * _ENTITY_DESCRIPTOR_WIDTH,
    )
    for index in range(count):
        base = index * _ENTITY_DESCRIPTOR_WIDTH
        kind = descriptors[base + _ED_KIND]
        name = descriptors[base + _ED_NAME]
        if descriptors[base + _ED_CAPACITY_MODE] == _CAPACITY_FIELD:
            address = self_addr + descriptors[base + _ED_CAPACITY]
            capacity = np.uint64(carray(
                _FLOAT64_FROM_ADDRESS(address), 1)[0])
        else:
            capacity = np.uint64(descriptors[base + _ED_CAPACITY])
        if kind == _ENTITY_BUFFER:
            handle = _b.buffer_create()
            _b.buffer_initialize(handle, name, capacity)
        elif kind == _ENTITY_RESOURCE:
            handle = _b.resource_create()
            _b.resource_initialize(handle, name)
        elif kind == _ENTITY_RESOURCEPOOL:
            handle = _b.resourcepool_create()
            _b.resourcepool_initialize(handle, name, capacity)
        elif kind == _ENTITY_OBJECTQUEUE:
            handle = _b.objectqueue_create()
            _b.objectqueue_initialize(handle, name, capacity)
        elif kind == _ENTITY_DATASET:
            handle = _b.dataset_create()
            _b.dataset_initialize(handle)
        elif kind == _ENTITY_CONDITION:
            handle = _b.condition_create()
            _b.condition_initialize(handle, name)
        else:
            handle = _b.priorityqueue_create()
            _b.priorityqueue_initialize(handle, name, capacity)
        target = carray(
            _INT64_FROM_ADDRESS(
                self_addr + descriptors[base + _ED_DEST_OFFSET]),
            1,
        )
        target[0] = handle


def _runtime_recording_event(subject, obj):
    """Start/stop recording or stop processes through entity descriptors."""
    env = carray(subject, 1)[0]
    if obj == 2:
        call_void_callback(env[_TRIAL_STOP_FIELD], subject)
        return
    self_addr = addressof(subject)
    count = env[_ENTITY_DESCRIPTOR_COUNT_FIELD]
    descriptors = carray(
        _INT64_FROM_ADDRESS(env[_ENTITY_DESCRIPTORS_FIELD]),
        max(1, count) * _ENTITY_DESCRIPTOR_WIDTH,
    )
    for index in range(count):
        base = index * _ENTITY_DESCRIPTOR_WIDTH
        kind = descriptors[base + _ED_KIND]
        handle = carray(
            _INT64_FROM_ADDRESS(
                self_addr + descriptors[base + _ED_DEST_OFFSET]),
            1,
        )[0]
        if obj == 0:
            if kind == _ENTITY_BUFFER:
                _b.buffer_recording_start(handle)
            elif kind == _ENTITY_RESOURCE:
                _b.resource_recording_start(handle)
            elif kind == _ENTITY_RESOURCEPOOL:
                _b.resourcepool_recording_start(handle)
            elif kind == _ENTITY_OBJECTQUEUE:
                _b.objectqueue_recording_start(handle)
            elif kind == _ENTITY_DATASET:
                _b.dataset_reset(handle)
            elif kind == _ENTITY_PRIORITYQUEUE:
                _b.priorityqueue_recording_start(handle)
        else:
            if kind == _ENTITY_BUFFER:
                _b.buffer_recording_stop(handle)
            elif kind == _ENTITY_RESOURCE:
                _b.resource_recording_stop(handle)
            elif kind == _ENTITY_RESOURCEPOOL:
                _b.resourcepool_recording_stop(handle)
            elif kind == _ENTITY_OBJECTQUEUE:
                _b.objectqueue_recording_stop(handle)
            elif kind == _ENTITY_PRIORITYQUEUE:
                _b.priorityqueue_recording_stop(handle)


def _runtime_trial_initialize(vtrl):
    """Initialize one trial using only the fixed lifecycle header."""
    env = carray(vtrl, 1)[0]
    self_addr = addressof(vtrl)
    _b.logger_apply_flags()
    _b.event_queue_initialize(env["start_time"])
    _b.random_initialize(env["seed"])
    timestamp = env["start_time"] + env["warmup_s"]
    _b.event_schedule(
        env[_RECORDING_EVENT_FIELD], self_addr, 0, timestamp, 0)
    timestamp += env["duration_s"]
    _b.event_schedule(
        env[_RECORDING_EVENT_FIELD], self_addr, 1, timestamp, 0)
    timestamp += env["cooldown_s"]
    _b.event_schedule(
        env[_RECORDING_EVENT_FIELD], self_addr, 2, timestamp, 0)


def _runtime_trial_teardown(vtrl):
    """Run collectors and destroy runtime state through stable tables."""
    env = carray(vtrl, 1)[0]
    call_void_callback(env[_TRIAL_COLLECT_FIELD], vtrl)
    call_void_callback(env[_TRIAL_PROCESS_CLEANUP_FIELD], vtrl)
    self_addr = addressof(vtrl)
    count = env[_ENTITY_DESCRIPTOR_COUNT_FIELD]
    descriptors = carray(
        _INT64_FROM_ADDRESS(env[_ENTITY_DESCRIPTORS_FIELD]),
        max(1, count) * _ENTITY_DESCRIPTOR_WIDTH,
    )
    for index in range(count):
        base = index * _ENTITY_DESCRIPTOR_WIDTH
        kind = descriptors[base + _ED_KIND]
        handle = carray(
            _INT64_FROM_ADDRESS(
                self_addr + descriptors[base + _ED_DEST_OFFSET]),
            1,
        )[0]
        if kind == _ENTITY_BUFFER:
            _b.buffer_destroy(handle)
        elif kind == _ENTITY_RESOURCE:
            _b.resource_destroy(handle)
        elif kind == _ENTITY_RESOURCEPOOL:
            _b.resourcepool_destroy(handle)
        elif kind == _ENTITY_OBJECTQUEUE:
            _b.objectqueue_destroy(handle)
        elif kind == _ENTITY_DATASET:
            _b.dataset_destroy(handle)
        elif kind == _ENTITY_CONDITION:
            _b.condition_destroy(handle)
        else:
            _b.priorityqueue_terminate(handle)
            _b.priorityqueue_destroy(handle)
    _b.event_queue_terminate()
    _b.random_terminate()


def _runtime_trial(vtrl):
    """Fixed trial dispatcher independent of every model record suffix."""
    env = carray(vtrl, 1)[0]
    call_void_callback(env[_TRIAL_INITIALIZE_FIELD], vtrl)
    call_void_callback(env[_TRIAL_ENTITIES_FIELD], vtrl)
    call_void_callback(env[_TRIAL_PROCESSES_FIELD], vtrl)
    _b.event_queue_execute()
    call_void_callback(env[_TRIAL_TEARDOWN_FIELD], vtrl)


class _LoadedCFunc:
    """A callback from a fork-compiled library loaded into the parent."""

    __slots__ = ("address", "library", "native_name", "serialized_state")

    def __init__(self, library, native_name: str, serialized_state=None):
        self.library = library
        self.native_name = native_name
        self.serialized_state = serialized_state
        self.address = library.get_pointer_to_function(native_name)


def _load_compiled_library(state):
    """Load one worker's linked object-code library into the parent."""
    from numba.core.registry import cpu_target
    from numba.core.runtime import nrt

    nrt.rtsys.initialize(cpu_target.target_context)
    return cpu_target.target_context.codegen().unserialize_library(state)


_PARALLEL_CFUNC_JOBS: tuple[tuple[Any, Callable[..., Any]], ...] = ()


def _compile_cfunc_job(index: int):
    """Compile and serialize one callback in a forked worker."""
    signature, function = _PARALLEL_CFUNC_JOBS[index]
    callback = _compile_parallel_cfunc(signature, function)
    return (
        index,
        callback._library.serialize_using_object_code(),
        callback.native_name,
    )


def _compile_cfunc_worker(connection) -> None:
    """Serve callback indexes over a pipe in a forked compiler process."""
    try:
        while True:
            index = connection.recv()
            if index is None:
                return
            try:
                connection.send((True, _compile_cfunc_job(index)))
            except BaseException as exc:
                connection.send((False, f"{type(exc).__name__}: {exc}"))
                return
    finally:
        connection.close()


def _compile_complexity(function: Callable[..., Any]) -> int:
    """Estimate callback cost, including lazy Numba callees it closes over."""
    seen: set[int] = set()

    def visit(candidate) -> int:
        py_func = getattr(candidate, "py_func", candidate)
        if not inspect.isfunction(py_func) or id(py_func) in seen:
            return 0
        seen.add(id(py_func))
        score = len(py_func.__code__.co_code)
        if py_func.__closure__ is not None:
            for cell in py_func.__closure__:
                score += visit(cell.cell_contents)
        for name in py_func.__code__.co_names:
            value = py_func.__globals__.get(name)
            if hasattr(value, "py_func"):
                score += visit(value)
        return score

    return visit(function)


def _callback_function_key(function: Callable[..., Any]) -> str:
    """Fingerprint lowered code and compile-time values used by AOT reuse."""
    digest = hashlib.sha256()
    seen: set[int] = set()

    def add(value) -> None:
        if id(value) in seen:
            return
        if inspect.isfunction(value) or hasattr(value, "py_func"):
            py_func = getattr(value, "py_func", value)
            if not inspect.isfunction(py_func):
                return
            seen.add(id(value))
            code = py_func.__code__
            digest.update(code.co_code)
            digest.update(repr(code.co_consts).encode())
            digest.update(repr(code.co_names).encode())
            digest.update(repr(py_func.__defaults__).encode())
            digest.update(repr(py_func.__kwdefaults__).encode())
            digest.update(
                repr(getattr(py_func, "__cimba_cache_salt__", None)).encode()
            )
            source = getattr(py_func, "__cimba_source__", None)
            if source is not None:
                digest.update(source.encode())
            if py_func.__closure__ is not None:
                for cell in py_func.__closure__:
                    add(cell.cell_contents)
            for name in code.co_names:
                if name in py_func.__globals__:
                    add(py_func.__globals__[name])
        elif isinstance(value, np.ndarray):
            seen.add(id(value))
            digest.update(value.dtype.str.encode())
            digest.update(repr(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, np.generic):
            digest.update(value.dtype.str.encode())
            digest.update(value.tobytes())
        elif isinstance(value, (tuple, list)):
            seen.add(id(value))
            for item in value:
                add(item)
        elif isinstance(value, Mapping):
            seen.add(id(value))
            for key in sorted(value, key=repr):
                add(key)
                add(value[key])
        elif isinstance(value, pytypes.ModuleType):
            digest.update(value.__name__.encode())
            digest.update(repr(getattr(value, "__version__", None)).encode())
        elif isinstance(value, (str, bytes, int, float, bool, type(None))):
            digest.update(repr(value).encode())
        elif inspect.isclass(value) and hasattr(value, "_dtype"):
            digest.update(repr(value._dtype.descr).encode())

    add(function)
    return digest.hexdigest()


_CALLBACK_CACHE_FORMAT = 3
_MEMORY_CALLBACK_CACHE: dict[str, Any] = {}
_MEMORY_CALLBACK_CACHE_LOCK = threading.RLock()
_CALLBACK_PLATFORM_KEY: str | None = None


@dataclass
class _CacheCounters:
    """Mutable counters shared by one compilation operation."""

    memory_hits: int = 0
    disk_hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def hits(self) -> int:
        return self.memory_hits + self.disk_hits


def _cache_enabled() -> bool:
    return (
        os.environ.get("CIMBA_CACHE", "1").lower()
        not in {"0", "false", "no", "off"}
    )


def _callback_cache_dir() -> Path:
    configured = os.environ.get("CIMBA_CACHE_DIR")
    if configured:
        return Path(configured)
    if sys.platform == "win32":
        root = Path(os.environ.get(
            "LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "cimba" / "Cache" / "callbacks"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "cimba" / "callbacks"
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "cimba" / "callbacks"


def _callback_cache_platform_key() -> str:
    """Version and target boundary for persisted native object code."""
    global _CALLBACK_PLATFORM_KEY
    with _MEMORY_CALLBACK_CACHE_LOCK:
        if _CALLBACK_PLATFORM_KEY is not None:
            return _CALLBACK_PLATFORM_KEY
        from llvmlite import binding as llvm
        from numba.core import config
        from numba.core.registry import cpu_target

        try:
            cimba_version = importlib.metadata.version("cimba")
        except importlib.metadata.PackageNotFoundError:
            cimba_version = "source-tree"
        values = (
            _CALLBACK_CACHE_FORMAT,
            cimba_version,
            numba.__version__,
            llvmlite.__version__,
            np.__version__,
            sys.implementation.cache_tag,
            sys.byteorder,
            platform.system(),
            platform.machine(),
            llvm.get_host_cpu_name(),
            llvm.get_host_cpu_features().flatten(),
            config.CPU_NAME,
            config.CPU_FEATURES,
            str(cpu_target.target_context.target_data),
        )
        _CALLBACK_PLATFORM_KEY = hashlib.sha256(
            repr(values).encode()).hexdigest()
        return _CALLBACK_PLATFORM_KEY


def _callback_cache_key(signature: Any, function: Callable[..., Any]) -> str:
    digest = hashlib.sha256()
    digest.update(_callback_cache_platform_key().encode())
    digest.update(repr(signature).encode())
    digest.update(_callback_function_key(function).encode())
    return digest.hexdigest()


def _callback_cache_path(key: str) -> Path:
    return _callback_cache_dir() / key[:2] / f"{key}.cimba"


def _load_cached_callback(key: str, counters: _CacheCounters) -> Any | None:
    if not _cache_enabled():
        counters.misses += 1
        return None
    with _MEMORY_CALLBACK_CACHE_LOCK:
        callback = _MEMORY_CALLBACK_CACHE.get(key)
    if callback is not None:
        counters.memory_hits += 1
        return callback
    try:
        path = _callback_cache_path(key)
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("format") != _CALLBACK_CACHE_FORMAT \
                or payload.get("key") != key:
            raise ValueError("incompatible callback cache entry")
        callback = _LoadedCFunc(
            _load_compiled_library(payload["state"]),
            payload["native_name"],
            payload["state"],
        )
    except Exception:
        # Files can be stale, truncated, or incompatible with a local LLVM
        # build despite their metadata. A cache miss must remain harmless.
        counters.misses += 1
        return None
    with _MEMORY_CALLBACK_CACHE_LOCK:
        _MEMORY_CALLBACK_CACHE[key] = callback
    counters.disk_hits += 1
    return callback


def _store_cached_callback(
    key: str,
    callback: Any,
    counters: _CacheCounters,
    *,
    serialized_state: Any | None = None,
) -> None:
    if not _cache_enabled():
        return
    with _MEMORY_CALLBACK_CACHE_LOCK:
        _MEMORY_CALLBACK_CACHE[key] = callback
    temporary: Path | None = None
    try:
        path = _callback_cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = serialized_state
        if state is None:
            state = getattr(callback, "serialized_state", None)
        if state is None:
            library = getattr(callback, "_library", None)
            if library is None:
                library = callback.library
            state = library.serialize_using_object_code()
        payload = {
            "format": _CALLBACK_CACHE_FORMAT,
            "key": key,
            "native_name": callback.native_name,
            "state": state,
        }
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{key}.", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
        counters.writes += 1
    except Exception:
        # A cache is an optimization only. Compilation has already succeeded.
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _compile_uncached_cfuncs(
    jobs: Sequence[tuple[Any, Callable[..., Any]]],
    *,
    warm_parent: bool = True,
) -> list[Any]:
    """Compile independent callbacks concurrently when ``fork`` is available."""
    if len(jobs) < 2:
        return [_compile_parallel_cfunc(signature, function)
                for signature, function in jobs]
    try:
        import multiprocessing

        # Forking from a non-main Python thread is unsafe, and the inherited
        # job table below is intentionally single-owner. Keep that uncommon
        # path functional with serial compilation.
        if threading.current_thread() is not threading.main_thread():
            raise ValueError

        context = multiprocessing.get_context("fork")
    except (ImportError, ValueError):
        return [_compile_parallel_cfunc(signature, function)
                for signature, function in jobs]

    # Populate Numba's registries and native runtime before forking.  These
    # tables are copy-on-write state, so every compiler worker can inherit the
    # setup instead of repeating it on its first callback.
    from numba.core.registry import cpu_target
    from numba.core.runtime import nrt
    cpu_target.target_context.refresh()
    nrt.rtsys.initialize(cpu_target.target_context)
    cpu_target.target_context.codegen()

    callbacks: list[Any] = [None] * len(jobs)
    use_warm_parent = warm_parent and len(jobs) > 4
    warm_index = -1
    warm_signature = None
    warm_function = None
    if use_warm_parent:
        # A cfunc's first compiler initialization dominates small callbacks.
        # Give the parent the cheapest job while workers take substantive work.
        warm_index = min(
            range(len(jobs)),
            key=lambda index: _compile_complexity(jobs[index][1]),
        )
        warm_signature, warm_function = jobs[warm_index]
        remaining = [
            index for index in range(len(jobs)) if index != warm_index]
    else:
        remaining = list(range(len(jobs)))
    # Start expensive-looking jobs first so the pipe scheduler can fill gaps
    # with short callbacks as workers finish.
    remaining.sort(
        key=lambda index: _compile_complexity(jobs[index][1]),
        reverse=True,
    )

    global _PARALLEL_CFUNC_JOBS
    _PARALLEL_CFUNC_JOBS = tuple(jobs[index] for index in remaining)
    # Cap concurrent LLVM instances to control memory and CPU contention. The
    # parent counts as the sixth compiler when it participates.
    worker_limit = 5 if use_warm_parent else 6
    worker_count = min(worker_limit, len(remaining))
    connections = []
    processes = []
    completed = False
    try:
        for _ in range(worker_count):
            parent_connection, child_connection = context.Pipe()
            process = context.Process(
                target=_compile_cfunc_worker,
                args=(child_connection,),
            )
            process.start()
            child_connection.close()
            connections.append(parent_connection)
            processes.append(process)

        next_job = worker_count
        for connection, index in zip(connections, range(worker_count)):
            connection.send(index)

        if use_warm_parent:
            assert warm_signature is not None and warm_function is not None
            callbacks[warm_index] = _compile_parallel_cfunc(
                warm_signature, warm_function)

        serialized = []
        active = set(connections)
        from multiprocessing.connection import wait
        while active:
            for connection in wait(active):
                ok, payload = connection.recv()
                if not ok:
                    raise RuntimeError(
                        f"parallel callback compilation failed: {payload}")
                serialized.append(payload)
                if next_job < len(remaining):
                    connection.send(next_job)
                    next_job += 1
                else:
                    connection.send(None)
                    active.remove(connection)
        completed = True
    finally:
        for connection in connections:
            connection.close()
        if not completed:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join()
        _PARALLEL_CFUNC_JOBS = ()
    for local_index, state, native_name in serialized:
        library = _load_compiled_library(state)
        callbacks[remaining[local_index]] = _LoadedCFunc(
            library, native_name, state)
    return callbacks


def _compile_cfuncs(
    jobs: Sequence[tuple[Any, Callable[..., Any]]],
    *,
    warm_parent: bool = True,
    cache_counters: _CacheCounters | None = None,
) -> list[Any]:
    """Load content-addressed callbacks and compile only cache misses."""
    if not jobs:
        return []
    counters = cache_counters if cache_counters is not None \
        else _CacheCounters()
    callbacks: list[Any] = [None] * len(jobs)
    missing_jobs: list[tuple[Any, Callable[..., Any]]] = []
    missing_keys: list[str] = []
    missing_indices: dict[str, list[int]] = {}
    for index, (signature, function) in enumerate(jobs):
        key = _callback_cache_key(signature, function)
        callback = _load_cached_callback(key, counters)
        if callback is None:
            indices = missing_indices.setdefault(key, [])
            indices.append(index)
            if len(indices) == 1:
                missing_jobs.append((signature, function))
                missing_keys.append(key)
        else:
            callbacks[index] = callback
    if missing_jobs:
        compiled = _compile_uncached_cfuncs(
            missing_jobs, warm_parent=warm_parent)
        for key, callback in zip(missing_keys, compiled):
            for index in missing_indices[key]:
                callbacks[index] = callback
            _store_cached_callback(key, callback, counters)
    return callbacks


def _native_names(names: Iterable[str]) -> dict[str, str]:
    """Return deterministic, unique names that fit a native name buffer."""
    result: dict[str, str] = {}
    used: set[str] = set()
    for name in names:
        encoded = name.encode("utf-8")
        if len(encoded) <= _NATIVE_NAME_BYTES and name not in used:
            candidate = name
        else:
            salt = 0
            while True:
                digest = hashlib.sha256(
                    f"{salt}:{name}".encode("utf-8")).hexdigest()[:10]
                suffix = f"__{digest}"
                budget = _NATIVE_NAME_BYTES - len(suffix)
                prefix = encoded[:budget].decode("utf-8", errors="ignore")
                candidate = f"{prefix}{suffix}"
                if candidate not in used:
                    break
                salt += 1
        result[name] = candidate
        used.add(candidate)
    return result


class Struct:
    """Per-process data fields, declared like a dataclass: subclass it
    and annotate the fields (``float`` or ``int``). A process function
    asks for its own view by annotating a final parameter with the
    subclass::

        class Visitor(sim.Struct):
            patience: float
            rides: int

        class Park(sim.Model):
            @sim.process(copies=4)
            def visitor(env, vip: Visitor):
                vip.patience = cimba.random.triangular(0.5, 1.0, 1.5)

    Each process copy then carries its own fields, zeroed at creation, in
    the same native allocation as the process (this is the Python form of
    the C tutorial's ``struct visitor { struct cmb_process core; ... }``).
    Subclassing a Struct subclass inherits its fields.

    Other processes reach the same fields through the process handle:
    inside model code, ``Visitor(handle)`` returns a read/write view --
    so a handle pulled from a queue is all a server needs to update a
    visitor's statistics. ``@sim.process(struct=Visitor)`` attaches the
    fields without the view parameter.
    """

    if TYPE_CHECKING:
        def __init__(self, process: Handle) -> None: ...

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        raise TypeError(f"{cls.__name__}(handle) views are only available "
                        "inside compiled model code")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        fields = []
        for fname, hint in class_type_hints(cls).items():
            if fname.startswith("_"):
                continue
            if hint is float:
                fields.append((fname, "<f8"))
            elif hint is int:
                fields.append((fname, "<i8"))
            else:
                raise TypeError(f"struct field '{fname}': only float and "
                                "int fields are supported")
        if not fields:
            raise ValueError(f"struct '{cls.__name__}' declares no fields")
        cls._dtype = np.dtype(fields)
        cls._alloc_size = _PROC_DATA_OFFSET + cls._dtype.itemsize

        cast = ptr_caster(from_dtype(cls._dtype))
        offset = _PROC_DATA_OFFSET

        def view(process):
            return carray(cast(process + offset), 1)[0]

        @_nb_overload(cls)
        def struct_view(process):
            if not isinstance(process, types.Integer):
                return None
            return view


def _is_struct_class(obj: Any) -> bool:
    return (isinstance(obj, type) and issubclass(obj, Struct)
            and obj is not Struct)


@dataclass
class _ProcDecl:
    """A lowered class-declared ``@sim.process`` callback."""

    name: str
    fn: Callable[..., Any]
    copies: int
    priority: int
    indexed: bool                  # takes the copy index argument
    struct: type[Struct] | None    # per-process fields, if any
    injected: bool                 # fn receives its own struct view
    spawnable: bool                # created by sim.spawn(), not at setup
    spawn_field: str | None = None # env spawn descriptor lands in
    spawn_index: int | None = None # shaped descriptor element, if any
    process_field: str | None = None # env Processes field handles land in
    process_offset: int = 0        # first handle slot for this process

    @property
    def alloc_size(self) -> int:
        return (self.struct._alloc_size if self.struct is not None
                else _PROC_DATA_OFFSET)


@dataclass(frozen=True)
class CompilationPlan:
    """Immutable class callback compilation work derived from a real model.

    The plan is created from the first normally initialized model instance,
    never from a partially initialized object constructed with
    ``object.__new__``.  It is exposed for diagnostics; callback functions and
    the owning model are retained privately so Numba lowering state stays
    alive for the resulting native libraries.
    """

    model_name: str
    callback_dtype: np.dtype
    process_names: tuple[str, ...]
    process_keys: tuple[str, ...]
    predicate_names: tuple[str, ...]
    predicate_keys: tuple[str, ...]
    event_names: tuple[str, ...]
    event_keys: tuple[str, ...]
    collect_keys: tuple[str, ...]
    lifecycle_key: tuple[str, ...]
    callback_count: int
    _record_type: Any
    _lifecycle_jobs: tuple[tuple[Any, Callable[..., Any]], ...]
    _owner: Any


@dataclass(frozen=True)
class CompilationStatus:
    """Observable state of a model class's reusable compilation plan."""

    state: str
    seconds: float = 0.0
    process_count: int = 0
    callback_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    error: str | None = None


@dataclass(frozen=True)
class CompilationCacheStats:
    """Cache activity for one model instance's remaining callbacks."""

    hits: int = 0
    misses: int = 0
    writes: int = 0


@dataclass(frozen=True)
class _CompiledCallbackPlan:
    plan: CompilationPlan
    procs: tuple[tuple[str, Any], ...]
    predicates: tuple[tuple[str, Any], ...]
    events: tuple[tuple[str, Any], ...]
    extras: tuple[Any, ...]


class _Compiled(TypedDict):
    """Artifacts of Model._compile(), kept alive for the model's lifetime.
    The callables are Numba cfunc/dispatcher objects (untyped upstream)."""

    trial: Any
    events: tuple[Any, ...]
    procs: dict[str, Any]
    preds: dict[str, Any]
    user_events: dict[str, Any]
    collect_callbacks: dict[int, Any]
    collect_descriptors: np.ndarray
    process_descriptors: np.ndarray
    process_handle_count: int
    entity_descriptors: np.ndarray
    entity_descriptor_count: int
    runtime_text_handles: np.ndarray
    #: per-spawn-descriptor arrays sim.spawn() reads:
    #: [cfunc address, name cstring, allocation size]
    spawns: dict[str, np.ndarray]
    #: (spawn descriptor field, optional shaped-field index, process name)
    #: assignments applied to the experiment table.
    spawn_assignments: tuple[tuple[str, int | None, str], ...]
    dtype: np.dtype


@dataclass(frozen=True)
class ComponentFieldSchema:
    """Public flattened-layout metadata for one component-owned field."""

    path: str
    flattened_name: str
    kind: str
    owners: tuple[int, ...]
    concrete_types: tuple[type[Component], ...]
    logical_count: int
    shape: tuple[int, ...] | None

    @property
    def packed(self) -> bool:
        """Whether the field omits logical component instances."""
        return self.owners != tuple(range(self.logical_count))


def _as_capacity_dict(value: _Capacities) -> dict[str, int | str | None]:
    """Normalize a `stores`/`pools` declaration: a list means unbounded
    capacity; dict values are an int capacity or a param name string."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {name: None for name in value}


def _as_trace_rows(value: Any, n_trials: int, name: str) -> list[np.ndarray]:
    """Normalize a Trace value into one contiguous float64 row per trial:
    a 1-D array is shared by every trial, a 2-D array maps row i to trial
    i, and a sequence of 1-D arrays gives ragged per-trial traces."""
    if not isinstance(value, np.ndarray):
        try:
            value = np.asarray(value, dtype=np.float64)
        except (ValueError, TypeError):
            # Ragged: a sequence of per-trial 1-D arrays
            rows = [np.ascontiguousarray(row, dtype=np.float64)
                    for row in value]
            if len(rows) != n_trials:
                raise ValueError(
                    f"trace '{name}': expected {n_trials} per-trial "
                    f"arrays (one per trial), got {len(rows)}") from None
            for row in rows:
                if row.ndim != 1:
                    raise ValueError(f"trace '{name}': per-trial arrays "
                                     "must be 1-D") from None
            return rows
    arr = np.ascontiguousarray(value, dtype=np.float64)
    if arr.ndim == 1:
        return [arr] * n_trials
    if arr.ndim == 2:
        if arr.shape[0] != n_trials:
            raise ValueError(
                f"trace '{name}': expected {n_trials} rows (one per "
                f"trial, design-point-major with replications innermost), "
                f"got {arr.shape[0]}")
        return list(arr)
    raise ValueError(f"trace '{name}': expected a 1-D array (shared), a "
                     "2-D array (row per trial), or a sequence of 1-D "
                     "arrays")


def _as_param_axis(
    value: Any,
    shape: tuple[int, ...] | None,
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if shape is None:
        return np.atleast_1d(arr).reshape(-1)
    if arr.ndim == 0:
        return np.full((1, *shape), float(arr), dtype=np.float64)
    if arr.shape == shape:
        return np.ascontiguousarray(arr.reshape((1, *shape)),
                                    dtype=np.float64)
    if arr.ndim == len(shape) + 1 and arr.shape[1:] == shape:
        return np.ascontiguousarray(arr, dtype=np.float64)
    raise ValueError(
        f"parameter '{name}': expected a scalar, shape {shape}, or "
        f"(n, {', '.join(str(dim) for dim in shape)}) design rows; "
        f"got shape {arr.shape}")


def _n_design_points(axes: Sequence[np.ndarray]) -> int:
    """Design points in the cross product of the parameter axes; a
    scalar parameter is a length-1 axis and contributes a factor of 1."""
    n_points = 1
    for axis in axes:
        n_points *= axis.shape[0]
    return n_points


def _trace_generator_wants_index(fn: Callable[..., ArrayLike]) -> bool:
    try:
        required = [p for p in inspect.signature(fn).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY,
                                  p.POSITIONAL_OR_KEYWORD)
                    and p.default is p.empty]
    except (TypeError, ValueError):
        return False
    return len(required) >= 2


def _draw_trial_seeds(seed: int | None, n_trials: int) -> np.ndarray:
    """The per-trial seed draw shared by experiment() and trial_seeds():
    one uint64 per trial seeds the in-sim RNG and, through trace_rng(),
    any callable trace generators."""
    rng = np.random.default_rng(
        seed if seed is not None else int(lib.cmb_random_hwseed()))
    return rng.integers(1, np.iinfo(np.uint64).max, size=n_trials,
                        dtype=np.uint64)


def trace_rng(trial_seed: int, field_name: str) -> np.random.Generator:
    """The generator a callable trace field sees for one trial: seeded
    from the trial's own cimba seed plus the field name, so a single
    experiment seed reproduces both the simulation streams and the
    generated traces, each trace field draws an independent stream, and
    any trial's trace can be regenerated post-hoc from its recorded
    ``exp["seed"]``.

    A callable with a ``trace_rng_name`` attribute uses that string
    instead of its field name -- callables sharing the tag receive
    identical per-trial generators, which lets one joint resample drive
    several trace fields with preserved cross-correlation."""
    tag = int.from_bytes(
        hashlib.sha256(field_name.encode()).digest()[:8], "little")
    return np.random.default_rng([int(trial_seed), tag])


def _generate_trace_rows(fn: Callable[..., ArrayLike], seeds: np.ndarray,
                         name: str) -> list[np.ndarray]:
    """Call a trace generator once per trial with that trial's
    trace_rng(); ``fn(rng)`` or ``fn(rng, trial_index)``."""
    wants_index = _trace_generator_wants_index(fn)
    rows: list[np.ndarray] = []
    tag = getattr(fn, "trace_rng_name", None) or name
    for i, s in enumerate(seeds):
        rng = trace_rng(int(s), tag)
        out = fn(rng, i) if wants_index else fn(rng)
        row = np.ascontiguousarray(out, dtype=np.float64)
        if row.ndim != 1:
            raise ValueError(f"trace '{name}': generator must return a "
                             f"1-D array, got {row.ndim}-D for trial {i}")
        rows.append(row)
    return rows


def _trace_rows_from_value(value: Any, seeds: np.ndarray, name: str,
                           n_trials: int) -> list[np.ndarray]:
    if callable(value):
        return _generate_trace_rows(value, seeds, name)
    return _as_trace_rows(value, n_trials, name)


def _trace_slot_name(name: str, index: int) -> str:
    return f"{name}[{index}]"


def _as_single_trace_row(value: Any, name: str, context: str) -> np.ndarray:
    if callable(value):
        raise ValueError(
            f"trace '{name}': callable values are not valid for {context}")
    row = np.ascontiguousarray(value, dtype=np.float64)
    if row.ndim != 1:
        raise ValueError(
            f"trace '{name}': {context} must be 1-D, got {row.ndim}-D")
    return row


def _trace_grid_shape_error(name: str, n_trials: int, slots: int) -> str:
    return (
        f"trace '{name}': expected a 1-D array shared by every trial and "
        f"component, a 2-D array with {slots} component rows or {n_trials} "
        f"trial rows, a 3-D array with shape ({n_trials}, {slots}, length), "
        "or a sequence of component trace values"
    )


def _as_trace_array_grid(value: Any, n_trials: int, slots: int,
                         name: str) -> list[list[np.ndarray]]:
    arr = np.ascontiguousarray(value, dtype=np.float64)
    if arr.ndim == 1:
        return [[arr for _ in range(slots)] for _ in range(n_trials)]
    if arr.ndim == 2:
        if arr.shape[0] == slots:
            rows = [np.ascontiguousarray(arr[i])
                    for i in range(slots)]
            return [[rows[j] for j in range(slots)]
                    for _ in range(n_trials)]
        if arr.shape[0] == n_trials:
            rows = [np.ascontiguousarray(arr[i])
                    for i in range(n_trials)]
            return [[rows[i] for _ in range(slots)]
                    for i in range(n_trials)]
        raise ValueError(_trace_grid_shape_error(name, n_trials, slots))
    if arr.ndim == 3:
        if arr.shape[0] != n_trials or arr.shape[1] != slots:
            raise ValueError(_trace_grid_shape_error(name, n_trials, slots))
        return [
            [np.ascontiguousarray(arr[i, j]) for j in range(slots)]
            for i in range(n_trials)
        ]
    raise ValueError(_trace_grid_shape_error(name, n_trials, slots))


def _as_trace_sequence_grid(value: Any, seeds: np.ndarray, n_trials: int,
                            slots: int, name: str) -> list[list[np.ndarray]]:
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(_trace_grid_shape_error(name, n_trials, slots)) \
            from exc

    if len(values) == slots:
        slot_rows = [
            _trace_rows_from_value(slot_value, seeds,
                                   _trace_slot_name(name, slot), n_trials)
            for slot, slot_value in enumerate(values)
        ]
        return [
            [slot_rows[slot][trial] for slot in range(slots)]
            for trial in range(n_trials)
        ]

    if len(values) == n_trials:
        rows: list[list[np.ndarray]] = []
        for trial, trial_value in enumerate(values):
            try:
                slot_values = list(trial_value)
            except TypeError as exc:
                raise ValueError(_trace_grid_shape_error(
                    name, n_trials, slots)) from exc
            if len(slot_values) != slots:
                raise ValueError(_trace_grid_shape_error(
                    name, n_trials, slots))
            rows.append([
                _as_single_trace_row(
                    slot_values[slot],
                    _trace_slot_name(name, slot),
                    f"trial {trial}, component {slot}",
                )
                for slot in range(slots)
            ])
        return rows

    raise ValueError(_trace_grid_shape_error(name, n_trials, slots))


def _generate_trace_grid(fn: Callable[..., ArrayLike], seeds: np.ndarray,
                         n_trials: int, slots: int,
                         name: str) -> list[list[np.ndarray]]:
    wants_index = _trace_generator_wants_index(fn)
    tag = getattr(fn, "trace_rng_name", None) or name
    rows: list[list[np.ndarray]] = []
    for trial, seed in enumerate(seeds):
        rng = trace_rng(int(seed), tag)
        out = fn(rng, trial) if wants_index else fn(rng)
        try:
            rows.append(_as_trace_array_grid(out, 1, slots, name)[0])
        except ValueError:
            try:
                slot_values = list(out)
            except TypeError as exc:
                raise ValueError(
                    f"trace '{name}': generator must return a 1-D array or "
                    f"{slots} component rows for trial {trial}") from exc
            if len(slot_values) != slots:
                raise ValueError(
                    f"trace '{name}': generator must return a 1-D array or "
                    f"{slots} component rows for trial {trial}")
            rows.append([
                _as_single_trace_row(
                    slot_values[slot],
                    _trace_slot_name(name, slot),
                    f"trial {trial}, component {slot}",
                )
                for slot in range(slots)
            ])
    return rows


def _as_trace_grid(value: Any, seeds: np.ndarray, n_trials: int,
                   slots: int, name: str) -> list[list[np.ndarray]]:
    if slots == 1:
        return [[row] for row in _trace_rows_from_value(
            value, seeds, name, n_trials)]
    if callable(value):
        return _generate_trace_grid(value, seeds, n_trials, slots, name)
    if isinstance(value, np.ndarray):
        return _as_trace_array_grid(value, n_trials, slots, name)
    try:
        return _as_trace_array_grid(value, n_trials, slots, name)
    except (ValueError, TypeError):
        return _as_trace_sequence_grid(value, seeds, n_trials, slots, name)


def _spawnable_slot_label(field: str, index: int | None) -> str:
    return field if index is None else f"{field}[{index}]"


def _runtime_text_expression(
    value: Any,
    register: Callable[[int], int],
    env_name: str,
) -> tuple[ast.expr, bool] | None:
    """Convert constants containing ``log_text`` handles to sidecar reads."""
    if isinstance(value, int) and not isinstance(value, bool):
        if _b.cstring_value(value) is not None:
            slot = register(value)
            return (
                ast.Call(
                    func=ast.Name(
                        id="_CIMBA_RUNTIME_TEXT_HANDLE", ctx=ast.Load()),
                    args=[
                        ast.Subscript(
                            value=ast.Name(
                                id=env_name, ctx=ast.Load()),
                            slice=ast.Constant(
                                value=_RUNTIME_TEXT_HANDLES_FIELD),
                            ctx=ast.Load(),
                        ),
                        ast.Constant(value=slot),
                    ],
                    keywords=[],
                ),
                True,
            )
        return ast.Constant(value=value), False
    if isinstance(value, (str, bytes, float, bool, type(None))):
        return ast.Constant(value=value), False
    if isinstance(value, tuple):
        items = [
            _runtime_text_expression(item, register, env_name)
            for item in value
        ]
        if any(item is None for item in items):
            return None
        converted = [item for item in items if item is not None]
        return (
            ast.Tuple(elts=[item[0] for item in converted], ctx=ast.Load()),
            any(item[1] for item in converted),
        )
    if isinstance(value, list):
        items = [
            _runtime_text_expression(item, register, env_name)
            for item in value
        ]
        if any(item is None for item in items):
            return None
        converted = [item for item in items if item is not None]
        return (
            ast.List(elts=[item[0] for item in converted], ctx=ast.Load()),
            any(item[1] for item in converted),
        )
    if isinstance(value, Mapping):
        keys = [
            _runtime_text_expression(key, register, env_name)
            for key in value
        ]
        values = [
            _runtime_text_expression(item, register, env_name)
            for item in value.values()
        ]
        if any(item is None for item in (*keys, *values)):
            return None
        converted_keys = [item for item in keys if item is not None]
        converted_values = [item for item in values if item is not None]
        return (
            ast.Dict(
                keys=[item[0] for item in converted_keys],
                values=[item[0] for item in converted_values],
            ),
            any(item[1] for item in (*converted_keys, *converted_values)),
        )
    return None


class _RuntimeTextHandleLowerer(ast.NodeTransformer):
    """Replace captured process-local text pointers with sidecar lookups."""

    def __init__(
        self,
        *,
        namespace: Mapping[str, Any],
        local_names: set[str],
        register: Callable[[int], int],
        env_name: str,
    ) -> None:
        self.namespace = namespace
        self.local_names = local_names
        self.register = register
        self.env_name = env_name
        self.changed = False

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if (not isinstance(node.ctx, ast.Load)
                or node.id in self.local_names
                or node.id not in self.namespace):
            return node
        converted = _runtime_text_expression(
            self.namespace[node.id], self.register, self.env_name)
        if converted is None or not converted[1]:
            return node
        self.changed = True
        return ast.copy_location(converted[0], node)


_ExperimentResultT = TypeVar(
    "_ExperimentResultT", default="ExperimentResults")


@dataclass(frozen=True)
class _ResultLeaf:
    family: str
    flattened_name: str


class _ResultNamespace:
    """Read-only attribute access over the model's structured results."""

    __slots__ = ("_experiment", "_entries", "_path")

    def __init__(
        self,
        experiment: "Experiment[Any]",
        entries: Mapping[str, "_ResultLeaf | _ResultNamespace"],
        path: str,
    ) -> None:
        object.__setattr__(self, "_experiment", experiment)
        object.__setattr__(self, "_entries", MappingProxyType(dict(entries)))
        object.__setattr__(self, "_path", path)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"result namespace '{self._path}' is read-only")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        entry = self._entries.get(name)
        if entry is None:
            raise AttributeError(
                f"unknown result '{self._path}.{name}'; available names: "
                f"{', '.join(sorted(self._entries)) or '<none>'}"
            )
        if isinstance(entry, _ResultNamespace):
            return entry
        return self._experiment._result_value(entry)

    def __dir__(self) -> list[str]:
        return sorted(set(object.__dir__(self)) | set(self._entries))

    def __repr__(self) -> str:
        names = ", ".join(sorted(self._entries))
        return f"<{type(self).__name__} {self._path}: {names}>"


class ExperimentResults(_ResultNamespace):
    """The model-shaped structured results retained by an experiment.

    Outputs, captured datasets, and captured histories share this one object
    tree. Users may parameterize ``Model`` with a Protocol describing the
    exact tree for IDE completion and static checking.
    """


def _insert_result_leaf(
    tree: dict[str, Any],
    parts: Sequence[str],
    leaf: _ResultLeaf,
) -> None:
    """Insert a path, retaining structural component branches on conflicts."""
    if not parts or any(not part for part in parts):
        return
    node = tree
    for part in parts[:-1]:
        existing = node.get(part)
        if isinstance(existing, _ResultLeaf):
            return
        if existing is None:
            existing = {}
            node[part] = existing
        node = existing
    final = parts[-1]
    if isinstance(node.get(final), dict):
        return
    node.setdefault(final, leaf)


def _component_result_path(schema: ComponentFieldSchema) -> tuple[str, ...]:
    return tuple(part for part in schema.path.replace("[]", "").split(".")
                 if part)


def _result_tree(experiment: "Experiment[Any]") -> dict[str, Any]:
    """Merge outputs and captures into the model's nested object tree."""
    model = experiment.model
    tree: dict[str, Any] = {}
    schemas = model.component_schema()
    if not isinstance(schemas, tuple):
        schemas = (schemas,)

    component_output_names: set[str] = set()
    for schema in schemas:
        if schema.kind != "output":
            continue
        component_output_names.add(schema.flattened_name)
        _insert_result_leaf(
            tree,
            _component_result_path(schema),
            _ResultLeaf("outputs", schema.flattened_name),
        )
    for name in model.outputs:
        if name not in component_output_names:
            _insert_result_leaf(
                tree, (name,), _ResultLeaf("outputs", name))

    def add_captures(
        family: str,
        names: Sequence[str],
        kinds: set[str],
    ) -> None:
        schema_paths: dict[str, list[tuple[str, ...]]] = {}
        for schema in schemas:
            if schema.kind in kinds:
                schema_paths.setdefault(schema.flattened_name, []).append(
                    _component_result_path(schema))
        for name in names:
            paths = schema_paths.get(name, [(name,)])
            for path in paths:
                _insert_result_leaf(tree, path, _ResultLeaf(family, name))

    add_captures(
        "datasets", experiment._dataset_capture_names, {"dataset"})
    add_captures(
        "histories", experiment._history_capture_names,
        {"queue", "resource", "pool", "store", "pqueues"})
    return tree


def _result_namespaces(
    experiment: "Experiment[Any]",
    tree: Mapping[str, Any],
    path: str,
) -> Mapping[str, _ResultLeaf | _ResultNamespace]:
    entries: dict[str, _ResultLeaf | _ResultNamespace] = {}
    for name, value in tree.items():
        if isinstance(value, dict):
            entries[name] = _ResultNamespace(
                experiment,
                _result_namespaces(experiment, value, f"{path}.{name}"),
                f"{path}.{name}",
            )
        else:
            entries[name] = value
    return entries


def _build_result_namespace(
    experiment: "Experiment[Any]",
) -> ExperimentResults:
    return ExperimentResults(
        experiment,
        _result_namespaces(experiment, _result_tree(experiment), "results"),
        "results",
    )


class Model(Generic[_ExperimentResultT]):
    """A simulation model. Subclass it and declare the env fields as
    annotations (Param, Output, Queue, Resource, Pool, Store, Dataset,
    Condition, State, Predicate) -- the subclass then types `env` in
    process bodies. Entity names may also be passed as keyword lists for
    quick callback-free untyped models. Class-declared model and component
    callbacks are compiled from the first real model instance and reused by
    the class."""

    # Standard trial-record fields, readable as env attributes in process
    # bodies (plain annotations, not declaration markers).
    start_time: float
    warmup_s: float
    duration_s: float
    cooldown_s: float
    seed: int

    _source: str
    __cimba_precompile__ = "eager"
    _cimba_callback_plan: CompilationPlan | None = None
    _cimba_callback_compiled: _CompiledCallbackPlan | None = None
    _cimba_callback_status = CompilationStatus("pending")
    _cimba_callback_lock = threading.RLock()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name, annotation in vars(cls).get("__annotations__", {}).items():
            if annotation is Spawnable:
                raise ValueError(
                    f"field '{name}': sim.Spawnable has been replaced by "
                    "@sim.process(spawnable=True)")
        mode = getattr(cls, "__cimba_precompile__", "eager")
        if mode not in {"eager", "lazy", "explicit"}:
            raise ValueError(
                "__cimba_precompile__ must be 'eager', 'lazy', or 'explicit'"
            )
        # Each subclass owns its plan, result, status, and synchronization.
        # No compilation occurs while the class body's module is importing.
        cls._cimba_callback_plan = None
        cls._cimba_callback_compiled = None
        cls._cimba_callback_status = CompilationStatus("pending")
        cls._cimba_callback_lock = threading.RLock()

    def __init__(self, name: str | None = None, *,
                 params: Iterable[str] = (),
                 outputs: Iterable[str] = (),
                 queues: _Capacities = None,
                 resources: Iterable[str] = (),
                 pools: _Capacities = None,
                 stores: _Capacities = None,
                 datasets: Iterable[str] = (),
                 conditions: Iterable[str] = (),
                 state: Iterable[str] = ()):
        decls = _class_declarations(type(self))
        self.name = name if name is not None else type(self).__name__
        for kind_name, names in (("param", params), ("output", outputs),
                                 ("resource", resources),
                                 ("dataset", datasets),
                                 ("condition", conditions),
                                 ("state", state)):
            for n in names:
                decls.add(_FieldDecl(n, _FIELD_KINDS[kind_name]))
        for kind_name, capacities in (("queue", queues), ("pool", pools),
                                      ("store", stores)):
            for n, cap in _as_capacity_dict(capacities).items():
                decls.add(_FieldDecl(n, _FIELD_KINDS[kind_name],
                                     capacity=cap))
        self._decls = decls

        # Backwards-compatible views of the declarations, by kind
        self.params = decls.names("param")
        self.param_defaults: dict[str, Any] = {
            field.name: field.default
            for field in decls.by_kind("param")
            if field.default is not _MISSING
        }
        self.outputs = decls.names("output")
        self.queues = {f.name: f.capacity for f in decls.by_kind("queue")}
        self.resources = decls.names("resource")
        self.pools = {f.name: f.capacity for f in decls.by_kind("pool")}
        self.stores = {f.name: f.capacity for f in decls.by_kind("store")}
        self.datasets = decls.names("dataset")
        self.conditions = decls.names("condition")
        self.state = decls.names("state")
        self.float_state: list[str] = decls.names("fstate")
        self.traces: list[str] = decls.names("trace")
        self.pqueues: dict[str, int] = {
            f.name: f.count for f in decls.by_kind("pqueues")}
        #: declared entity field name -> native binding prefix, for fields
        #: whose ``.history`` compiles to a native timeseries lookup.
        self.history_fields: dict[str, str] = {
            f.name: f.kind.binding
            for f in decls.by_kind("queue", "resource", "pool", "store",
                                   "pqueues")}
        self._history_captures: dict[str, HistoryCaptureSpec] = {}
        self._dataset_captures: dict[str, HistoryCaptureSpec] = {}
        #: declared entity field name -> field kind, for fields whose
        #: ``env.<entity>.method(...)`` calls (put/acquire/signal/...)
        #: compile to native helper calls.
        self.entity_fields: dict[str, str] = {
            f.name: f.kind.name
            for f in decls.by_kind("queue", "resource", "pool", "store",
                                   "pqueues", "condition", "event")}
        self._predicate_fields: list[str] = decls.names("predicate")
        self._event_fields: list[str] = decls.names("event")
        self._process_fields: list[str] = decls.names("processes")
        self._spawnable_fields: list[str] = decls.names("spawnable")
        self._component_decls: list[_ComponentDecl] = decls.components
        self._component_collection_decls: list[_ComponentDecl] = \
            decls.component_collections
        self._field_shapes: dict[str, tuple[int, ...]] = {
            f.name: f.shape for f in decls.fields.values()
            if f.shape is not None}
        # Component-owned entity fields retain a collection axis even when
        # the flattened dtype is scalar (a one-item collection).  The
        # authoring path's ``[]`` marker is the stable indication that a
        # field belongs to a component collection.
        self._indexed_history_fields: dict[str, int] = {}
        for root in self._component_roots.values():
            for decl in root.walk():
                if "[]" not in decl.item_display_name:
                    continue
                for field_decl in decl.decls.fields.values():
                    flat_name = decl.direct_field_map[field_decl.name]
                    if flat_name not in self.history_fields:
                        continue
                    count = len(decl.field_owners[field_decl.name])
                    self._indexed_history_fields[flat_name] = max(
                        count,
                        self._indexed_history_fields.get(flat_name, 0),
                    )
        self._components: dict[str, Component] = {}
        self._component_collections: dict[str, tuple[Component, ...]] = {}
        self._component_bindings: dict[str, tuple[Component, ...]] = {}
        self._component_spawnable_fields = {
            decl.direct_field_map[name]
            for root in self._component_roots.values()
            for decl in root.walk()
            for name in decl.decls.names("spawnable")
        }

        seen: set[str] = set()
        for field in decls.fields.values():
            _check_name(field.name, field.kind.name)
            seen.add(field.name)
        for root in self._component_roots.values():
            label = ("component collection" if root.collection
                     else "component")
            _check_name(root.name, label)
            if root.name in seen:
                raise ValueError(f"duplicate field name '{root.name}'")
            seen.add(root.name)
        for field in decls.by_kind("queue", "pool", "store"):
            cap = field.capacity
            if cap is not None and not isinstance(cap, int) \
                    and decls.kind_of(cap) != "param":
                raise ValueError(f"capacity '{cap}' is neither an int nor "
                                 "a declared param")
        self._seen = seen
        self._processes: list[_ProcDecl] = []
        # (name, fn, env field holding the compiled address)
        self._predicates: list[tuple[str, Callable[..., Any], str]] = []
        # (name, fn, env field holding the compiled address, takes_data)
        self._events: list[tuple[str, Callable[..., Any], str, bool]] = []
        self._collect: Callable[..., Any] | None = None
        # (lowered collect, instance count); count > 1 collects take the
        # instance index as their second argument
        self._component_collects: list[tuple[Callable[..., Any], int]] = []
        self._compiled: _Compiled | None = None
        self._callback_cache_stats = CompilationCacheStats()
        self._runtime_text_handles: list[int] = []
        self._runtime_text_slots: dict[str, int] = {}
        self._component_functions = _build_component_functions(
            self._component_roots.values())
        self._bind_components()
        self._register_component_processes()
        self._register_model_callbacks()
        self._class_process_count = len(self._processes)
        if type(self).__cimba_precompile__ == "eager":
            self._ensure_class_precompiled()

    def _bind_components(self) -> None:
        for decl in self._component_decls:
            component = copy.copy(decl.instances[0])
            self._components[decl.name] = component
            self._component_bindings[decl.name] = (component,)
            setattr(self, decl.name, component)
            self._bind_component_metadata(component, decl.name)
            self._bind_component_children(decl, (component,))
        for decl in self._component_collection_decls:
            components = tuple(copy.copy(template)
                               for template in decl.instances)
            self._component_collections[decl.name] = components
            self._component_bindings[decl.name] = components
            setattr(self, decl.name, list(components))
            for index, component in enumerate(components):
                self._bind_component_metadata(
                    component, f"{decl.name}[{index}]",
                    collection=decl.name, index=index)
            self._bind_component_children(decl, components)

    @classmethod
    def compilation_status(cls) -> CompilationStatus:
        """Return reusable class-callback compilation state."""
        return cls.__dict__.get(
            "_cimba_callback_status", CompilationStatus("pending"))

    def callback_cache_stats(self) -> CompilationCacheStats:
        """Return cache activity from this instance's latest compilation."""
        return self._callback_cache_stats

    @classmethod
    def compilation_plan(cls) -> CompilationPlan | None:
        """Return the immutable plan built from the first real instance."""
        return cls.__dict__.get("_cimba_callback_plan")

    @classmethod
    def precompile(cls, *args: Any, **kwargs: Any) -> CompilationStatus:
        """Explicitly prepare reusable class-declared callbacks.

        ``args`` and ``kwargs`` construct a normal model instance, so custom
        subclass initialization is honored.  A previous failed attempt is
        retried, which supports modules whose callback globals are populated
        after the model class declaration.
        """
        model = cls(*args, **kwargs)
        model._ensure_class_precompiled(retry=True)
        return cls.compilation_status()

    def _build_callback_compilation_plan(self) -> CompilationPlan | None:
        count = self._class_process_count
        if (count == 0 and not self._predicates and not self._events
                and not self._collects):
            return None
        callback_dtype = self.dtype
        rec = from_dtype(callback_dtype)
        trial_ptr = types.CPointer(rec)
        lifecycle_jobs = (
            (types.void(_LIFECYCLE_ABI_PTR, types.intp),
             _runtime_recording_event),
            (types.void(_LIFECYCLE_ABI_PTR),
             _runtime_trial_initialize),
            (types.void(_LIFECYCLE_ABI_PTR), _runtime_trial_entities),
            (types.void(_LIFECYCLE_ABI_PTR), _runtime_trial_processes),
            (types.void(_LIFECYCLE_ABI_PTR), _runtime_trial_teardown),
            (types.void(_LIFECYCLE_ABI_PTR), _runtime_trial),
            (types.void(_LIFECYCLE_ABI_PTR), _runtime_trial_stop),
            (types.void(_LIFECYCLE_ABI_PTR),
             _runtime_trial_process_cleanup),
            (types.void(_LIFECYCLE_ABI_PTR), _runtime_trial_collect),
        )
        processes = self._processes[:count]
        collect_jobs = tuple(
            (
                types.void(trial_ptr),
                self._direct_collect_callback(fn, index, instances),
            )
            for index, (fn, instances) in enumerate(
                self._collects)
        )
        return CompilationPlan(
            model_name=self.name,
            callback_dtype=callback_dtype,
            process_names=tuple(process.name for process in processes),
            process_keys=tuple(
                _callback_function_key(process.fn) for process in processes),
            predicate_names=tuple(name for name, _fn, _field
                                  in self._predicates),
            predicate_keys=tuple(
                f"{field}:{_callback_function_key(fn)}"
                for _name, fn, field in self._predicates),
            event_names=tuple(name for name, _fn, _field, _takes_data
                              in self._events),
            event_keys=tuple(
                f"{field}:{takes_data}:{_callback_function_key(fn)}"
                for _name, fn, field, takes_data in self._events),
            collect_keys=tuple(
                f"{_callback_function_key(fn)}:{instances}"
                for fn, instances in self._collects
            ),
            lifecycle_key=self._aot_lifecycle_key(),
            callback_count=(count + len(lifecycle_jobs)
                            + len(self._predicates) + len(self._events)
                            + len(collect_jobs)),
            _record_type=rec,
            _lifecycle_jobs=(*lifecycle_jobs, *collect_jobs),
            _owner=self,
        )

    def _ensure_class_precompiled(self, *, retry: bool = False) -> None:
        cls = type(self)
        with cls._cimba_callback_lock:
            if cls._cimba_callback_compiled is not None:
                return
            if cls._cimba_callback_status.state == "failed" and not retry:
                return
            started = time.perf_counter()
            counters = _CacheCounters()
            try:
                plan = self._build_callback_compilation_plan()
                cls._cimba_callback_plan = plan
                if plan is None:
                    cls._cimba_callback_status = CompilationStatus(
                        "unavailable",
                        seconds=time.perf_counter() - started,
                    )
                    return
                owner = plan._owner
                procs, preds, events, extras = \
                    owner._compile_callbacks(
                        plan._record_type,
                        plan._lifecycle_jobs,
                        warm_parent=True,
                        cache_counters=counters,
                        processes=owner._processes[
                            :owner._class_process_count],
                        predicates=owner._predicates,
                        events=owner._events,
                    )
                compiled = _CompiledCallbackPlan(
                    plan,
                    tuple(procs.items()),
                    tuple(preds.items()),
                    tuple(events.items()),
                    tuple(extras),
                )
                cls._cimba_callback_compiled = compiled
                cls._cimba_callback_status = CompilationStatus(
                    "ready",
                    seconds=time.perf_counter() - started,
                    process_count=len(plan.process_names),
                    callback_count=plan.callback_count,
                    cache_hits=counters.hits,
                    cache_misses=counters.misses,
                    cache_writes=counters.writes,
                )
            except Exception as exc:
                # Compilation still falls back to the instance's first
                # experiment, but the reason is now inspectable instead of
                # being silently discarded during class creation.
                cls._cimba_callback_compiled = None
                cls._cimba_callback_status = CompilationStatus(
                    "failed",
                    seconds=time.perf_counter() - started,
                    process_count=self._class_process_count,
                    cache_hits=counters.hits,
                    cache_misses=counters.misses,
                    cache_writes=counters.writes,
                    error=f"{type(exc).__name__}: {exc}",
                )

    def _aot_class_callbacks(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[int, Any]]:
        compiled = type(self).__dict__.get("_cimba_callback_compiled")
        if compiled is None:
            return {}, {}, {}, {}
        plan = compiled.plan
        count = self._class_process_count
        if tuple(process.name for process in self._processes[:count]) \
                != plan.process_names:
            return {}, {}, {}, {}
        if tuple(
            _callback_function_key(process.fn)
            for process in self._processes[:count]
        ) != plan.process_keys:
            return {}, {}, {}, {}
        if tuple(
            f"{field}:{_callback_function_key(fn)}"
            for _name, fn, field in self._predicates
        ) != plan.predicate_keys:
            return {}, {}, {}, {}
        if tuple(
            f"{field}:{takes_data}:{_callback_function_key(fn)}"
            for _name, fn, field, takes_data in self._events
        ) != plan.event_keys:
            return {}, {}, {}, {}
        if tuple(
            f"{_callback_function_key(fn)}:{instances}"
            for fn, instances in self._collects
        ) != plan.collect_keys:
            return {}, {}, {}, {}
        if self.dtype != plan.callback_dtype:
            return {}, {}, {}, {}
        extras = (dict(enumerate(compiled.extras))
                  if self._aot_lifecycle_key() == plan.lifecycle_key else {})
        return (dict(compiled.procs), dict(compiled.predicates),
                dict(compiled.events), extras)

    def _bind_component_metadata(
        self,
        component: Component,
        name: str,
        *,
        collection: str | None = None,
        index: int | None = None,
    ) -> None:
        try:
            component._cimba_model = self
            component._cimba_name = name
            if collection is not None:
                component._cimba_collection = collection
            if index is not None:
                component._cimba_index = index
        except AttributeError:
            pass

    def _bind_component_children(
        self,
        decl: _ComponentDecl,
        parents: tuple[Component, ...],
    ) -> None:
        for child in decl.children:
            bound: list[Component] = []
            if child.collection:
                for parent_index, parent in enumerate(parents):
                    start = child.parent_offsets[parent_index]
                    length = child.parent_lengths[parent_index]
                    items: list[Component] = []
                    for item_index in range(length):
                        child_index = start + item_index
                        component = copy.copy(child.instances[child_index])
                        bound.append(component)
                        items.append(component)
                        self._bind_component_metadata(
                            component,
                            child.process_names[child_index],
                            collection=child.name,
                            index=child_index,
                        )
                    setattr(parent, child.local_name, items)
            else:
                for parent_index, parent in enumerate(parents):
                    child_index = (child.parent_slots[parent_index]
                                   if child.parent_slots else parent_index)
                    if child_index < 0:
                        continue
                    component = copy.copy(child.instances[child_index])
                    bound.append(component)
                    setattr(parent, child.local_name, component)
                    self._bind_component_metadata(
                        component, child.process_names[child_index])
            bound_tuple = tuple(bound)
            self._component_bindings[child.name] = bound_tuple
            self._bind_component_children(child, bound_tuple)

    def _register_component_processes(self) -> None:
        for root in self._component_roots.values():
            for decl in root.walk():
                self._register_component_decl_processes(decl)
        self._component_collects = [
            (self._lower_runtime_text_handles(fn), count)
            for fn, count in self._component_collects
        ]

    def _register_model_callbacks(self) -> None:
        """Register the callbacks declared on this model class."""
        cls = type(self)
        protected = {
            name for name in vars(Model)
            if not name.startswith("_")
        }

        def methods(marker: str, kind: str):
            return _marked_methods(cls, marker, kind, stop=Model)

        process_methods = methods(_PROCESS_ATTR, "process")
        predicate_methods = methods(_PREDICATE_ATTR, "predicate")
        event_methods = methods(_EVENT_ATTR, "event")
        collect_methods = methods(_COLLECT_ATTR, "collect")

        all_methods = {
            **{name: "process" for name in process_methods},
            **{name: "predicate" for name in predicate_methods},
            **{name: "event" for name in event_methods},
            **{name: "collect" for name in collect_methods},
        }
        for name, kind in all_methods.items():
            if name in protected:
                raise ValueError(
                    f"model {kind} callback '{cls.__name__}.{name}' "
                    "shadows a public sim.Model operation")
            if name in self._seen:
                raise ValueError(
                    f"model {kind} callback '{cls.__name__}.{name}' "
                    "collides with a declared field; use a differently "
                    "named callback and field= when binding callback fields")

        # Predicates and events publish fields that model processes may
        # reference, so establish those bindings before lowering processes.
        for _name, (fn, spec) in predicate_methods.items():
            self._register_predicate(fn, target_field=spec.field)
        for _name, (fn, spec) in event_methods.items():
            self._register_event(fn, target_field=spec.field)

        for name, (fn, spec) in process_methods.items():
            if type(spec.copies) is not int:
                raise TypeError(
                    f"model process '{cls.__name__}.{name}' copies must be "
                    "an int")
            process_field = spec.field
            if process_field is not None \
                    and process_field not in self._process_fields:
                raise ValueError(
                    f"model process '{cls.__name__}.{name}' field= must name "
                    "a declared sim.Processes field")
            self._register_process(
                fn, copies=spec.copies, priority=spec.priority,
                struct=spec.struct, spawnable=spec.spawnable,
                _process_field=process_field, _owner_cls=cls)

        if len(collect_methods) > 1:
            names = ", ".join(collect_methods)
            raise ValueError(
                f"model '{cls.__name__}' has multiple collect callbacks: "
                f"{names}; override an inherited collector by using the "
                "same method name")
        for _name, (fn, _marker) in collect_methods.items():
            self._register_collect(fn)

    @staticmethod
    def _lower_shared_or_per_instance(
        decl: _ComponentDecl,
        lower_shared: Callable[[], Any],
        lower_instance: Callable[[int], Any],
    ) -> list[tuple[Any, int | None]]:
        """Compile a component method once for the whole decl, taking the
        copy index at runtime; index is None in the returned pairs. When
        the shared lowering fails -- per-instance Ref targets that cannot
        share one body -- fall back to one specialized function per
        instance (which either succeeds or reproduces the real error)."""
        if decl.count > 1:
            try:
                return [(lower_shared(), None)]
            except ValueError:
                pass
        return [(lower_instance(index), index) for index in range(decl.count)]

    def _register_component_decl_processes(
        self, decl: _ComponentDecl,
    ) -> None:
        """Lower and register one decl's process and collect methods.
        Spawnable process methods are always per-instance -- the spawn descriptor
        is what identifies the instance at runtime."""
        components = self._component_bindings[decl.name]
        if decl.polymorphic:
            for variant, indices in enumerate(
                    decl.specialization_groups()):
                cls = decl.class_at(indices[0])
                group_name = f"{decl.name}__variant_{variant}"
                for method_name, method, spec in \
                        _component_process_methods(cls):
                    counts = tuple(
                        spec.resolve_copies(
                            components[index],
                            f"{decl.process_names[index]}.{method_name}")
                        for index in indices)
                    representative = indices[0]
                    field_kind = (
                        decl.decls.kind_of(method_name)
                        if (method_name in decl.field_owners
                            and representative
                            in decl.field_owners[method_name])
                        else None)
                    spawnable = spec.spawnable or field_kind == "spawnable"
                    # Descriptor and handle slots are per logical instance;
                    # retain per-instance registration for these two field
                    # kinds while grouping ordinary scheduled processes.
                    if spawnable or field_kind == "processes":
                        for position, index in enumerate(indices):
                            lowered = _lower_component_process(
                                decl.process_names[index], decl, method_name,
                                method, _is_struct_class,
                                instance_index=index,
                                model_dataset_fields=self.datasets,
                                model_history_fields=self.history_fields,
                                model_entity_fields=self.entity_fields,
                                component_functions=self._component_functions)
                            spawn_field = (
                                decl.direct_field_map[method_name]
                                if spawnable else None)
                            spawn_index = None
                            if spawn_field is not None:
                                owners = decl.field_owners[method_name]
                                if len(owners) > 1:
                                    spawn_index = \
                                        decl.field_slots[method_name][index]
                            process_field = (
                                decl.direct_field_map[method_name]
                                if field_kind == "processes" else None)
                            process_offset = (
                                decl.process_offsets[method_name][index]
                                if process_field is not None else 0)
                            self._register_process(
                                lowered, copies=counts[position],
                                priority=spec.priority,
                                _spawn_field=spawn_field,
                                _spawn_index=spawn_index,
                                _process_field=process_field,
                                _process_offset=process_offset)
                        continue
                    if len(indices) == 1:
                        lowered = _lower_component_process(
                            group_name, decl, method_name, method,
                            _is_struct_class, instance_index=indices[0],
                            model_dataset_fields=self.datasets,
                            model_history_fields=self.history_fields,
                            model_entity_fields=self.entity_fields,
                            component_functions=self._component_functions)
                    else:
                        lowered = _lower_component_process(
                            group_name, decl, method_name, method,
                            _is_struct_class, copies_per_instance=counts,
                            instance_indices=indices,
                            model_dataset_fields=self.datasets,
                            model_history_fields=self.history_fields,
                            model_entity_fields=self.entity_fields,
                            component_functions=self._component_functions)
                    self._register_process(
                        lowered, copies=sum(counts),
                        priority=spec.priority)
                for method_name, method in _component_collect_methods(cls):
                    if len(indices) == 1:
                        lowered = _lower_component_collect(
                            group_name, decl, method_name, method,
                            instance_index=indices[0],
                            model_dataset_fields=self.datasets,
                            model_history_fields=self.history_fields,
                            model_entity_fields=self.entity_fields,
                            component_functions=self._component_functions)
                    else:
                        lowered = _lower_component_collect(
                            group_name, decl, method_name, method,
                            per_class=True, instance_indices=indices,
                            model_dataset_fields=self.datasets,
                            model_history_fields=self.history_fields,
                            model_entity_fields=self.entity_fields,
                            component_functions=self._component_functions)
                    self._component_collects.append(
                        (lowered, len(indices)))
            return

        for method_name, method, spec in _component_process_methods(decl.cls):
            counts = tuple(
                spec.resolve_copies(
                    component, f"{decl.process_names[index]}.{method_name}")
                for index, component in enumerate(components))
            field_kind = decl.decls.kind_of(method_name)
            spawnable = spec.spawnable or field_kind == "spawnable"

            def lower_instance(index: int) -> Any:
                return _lower_component_process(
                    decl.process_names[index], decl, method_name, method,
                    _is_struct_class, instance_index=index,
                    model_dataset_fields=self.datasets,
                    model_history_fields=self.history_fields,
                    model_entity_fields=self.entity_fields,
                    component_functions=self._component_functions)

            if spawnable:
                spawn_field = decl.direct_field_map[method_name]
                for index in range(decl.count):
                    self._register_process(
                        lower_instance(index), copies=counts[index],
                        priority=spec.priority, _spawn_field=spawn_field,
                        _spawn_index=index if decl.count > 1 else None)
                continue

            process_field = (decl.direct_field_map[method_name]
                             if field_kind == "processes" else None)
            lowered = self._lower_shared_or_per_instance(
                decl,
                lambda: _lower_component_process(
                    decl.name, decl, method_name, method, _is_struct_class,
                    copies_per_instance=counts,
                    model_dataset_fields=self.datasets,
                    model_history_fields=self.history_fields,
                    model_entity_fields=self.entity_fields,
                    component_functions=self._component_functions),
                lower_instance)
            for fn, index in lowered:
                if index is None:
                    copies, offset = sum(counts), 0
                else:
                    copies = counts[index]
                    offset = (decl.process_offsets[method_name][index]
                              if process_field is not None else 0)
                self._register_process(
                    fn, copies=copies, priority=spec.priority,
                    _process_field=process_field, _process_offset=offset)

        for method_name, method in _component_collect_methods(decl.cls):
            lowered = self._lower_shared_or_per_instance(
                decl,
                lambda: _lower_component_collect(
                    decl.name, decl, method_name, method, per_class=True,
                    model_dataset_fields=self.datasets,
                    model_history_fields=self.history_fields,
                    model_entity_fields=self.entity_fields,
                    component_functions=self._component_functions),
                lambda index: _lower_component_collect(
                    decl.process_names[index], decl, method_name, method,
                    instance_index=index,
                    model_dataset_fields=self.datasets,
                    model_history_fields=self.history_fields,
                    model_entity_fields=self.entity_fields,
                    component_functions=self._component_functions))
            for fn, index in lowered:
                self._component_collects.append(
                    (fn, decl.count if index is None else 1))

    @property
    def _component_roots(self) -> dict[str, _ComponentDecl]:
        return {decl.name: decl
                for decl in (*self._component_decls,
                             *self._component_collection_decls)}

    def component_schema(
        self,
        path: str | None = None,
    ) -> "ComponentFieldSchema | tuple[ComponentFieldSchema, ...]":
        """Describe component-owned flattened fields and packed ownership.

        ``path`` accepts the authoring path (with or without ``[]`` markers)
        or the flattened field name. With no path, returns every component
        field in declaration order.
        """
        schemas: list[ComponentFieldSchema] = []
        for root in self._component_roots.values():
            for decl in root.walk():
                for field_decl in decl.decls.fields.values():
                    name = field_decl.name
                    owners = decl.field_owners[name]
                    flat_name = decl.direct_field_map[name]
                    schemas.append(ComponentFieldSchema(
                        path=f"{decl.item_display_name}.{name}",
                        flattened_name=flat_name,
                        kind=field_decl.kind.name,
                        owners=owners,
                        concrete_types=tuple(
                            decl.instance_classes[index]
                            for index in owners),
                        logical_count=decl.count,
                        shape=self._field_shapes.get(flat_name),
                    ))
        if path is None:
            return tuple(schemas)
        normalized = path.replace("[]", "")
        matches = [
            schema for schema in schemas
            if (path == schema.flattened_name
                or normalized == schema.path.replace("[]", ""))
        ]
        if not matches:
            raise KeyError(f"unknown component field: {path}")
        if len(matches) > 1:
            raise KeyError(f"ambiguous component field: {path}")
        return matches[0]

    def _lower_component_refs(self, fn: _F) -> _F:
        return _lower_model_component_refs(
            fn, model_name=self.name,
            component_roots=self._component_roots,
            component_functions=self._component_functions,
        )

    def _lower_dataset_methods(self, fn: _F) -> _F:
        return _lower_dataset_methods(
            fn,
            model_name=self.name,
            dataset_fields=self.datasets,
        )

    def _lower_history_methods(self, fn: _F) -> _F:
        return _lower_history_methods(
            fn,
            model_name=self.name,
            history_fields=self.history_fields,
        )

    def _next_capture_slot(self) -> int:
        return sum(spec.slot_count for spec in (
            *self._history_captures.values(),
            *self._dataset_captures.values(),
        ))

    def _register_history_capture(
        self,
        name: str,
        binding: str,
        indexed_count: int | None = None,
    ) -> int:
        spec = self._history_captures.get(name)
        if spec is not None:
            if ((spec.shape is None) != (indexed_count is None)
                    or (spec.shape is not None
                        and spec.shape != (indexed_count,))):
                raise ValueError(
                    f"history field '{name}' is captured with inconsistent "
                    "indexing")
            return spec.slot
        if indexed_count is not None and indexed_count < 1:
            raise ValueError(
                f"indexed history field '{name}' has no collection items")
        slot = self._next_capture_slot()
        self._history_captures[name] = HistoryCaptureSpec(
            name=name,
            binding=binding,
            slot=slot,
            columns=3,
            shape=(indexed_count,) if indexed_count is not None else None,
        )
        return slot

    def _register_dataset_capture(self, name: str, binding: str) -> int:
        spec = self._dataset_captures.get(name)
        if spec is not None:
            return spec.slot
        slot = self._next_capture_slot()
        self._dataset_captures[name] = HistoryCaptureSpec(
            name=name,
            binding=binding,
            slot=slot,
            columns=1,
            shape=None,
        )
        return slot

    def _lower_history_capture_methods(self, fn: _F) -> _F:
        return lower_history_capture_methods(
            fn,
            model_name=self.name,
            history_fields=self.history_fields,
            indexed_history_fields=self._indexed_history_fields,
            register=self._register_history_capture,
        )

    def _lower_dataset_capture_methods(self, fn: _F) -> _F:
        return lower_dataset_capture_methods(
            fn,
            model_name=self.name,
            dataset_fields=set(self.datasets),
            register=self._register_dataset_capture,
        )

    def _entity_fields_with_hidden_events(self) -> dict[str, str]:
        """``self.entity_fields`` plus hidden ``_ev_<name>`` fields for
        unbound ``@sim.event`` callbacks (mirrors how
        ``process_dag()``'s ``entity_kinds`` gets the same merge)."""
        fields = dict(self.entity_fields)
        for _name, _fn, field, _takes_data in self._events:
            fields.setdefault(field, "event")
        return fields

    def _lower_entity_methods(
        self, fn: _F, *, extra_fields: Mapping[str, str] | None = None,
    ) -> _F:
        fields = self._entity_fields_with_hidden_events()
        if extra_fields:
            fields = {**fields, **extra_fields}
        return _lower_entity_methods(
            fn,
            model_name=self.name,
            entity_fields=fields,
        )

    def _lower_random_calls(self, fn: _F) -> _F:
        return lower_random_calls_in_function(
            fn,
            label=f"model '{self.name}' callback '{fn.__qualname__}'",
        )

    def _register_runtime_text_handle(self, address: int) -> int:
        """Register one local cstring address under a stable text slot."""
        text = _b.cstring_value(address)
        if text is None:
            raise ValueError("runtime text handle is not owned by Cimba")
        slot = self._runtime_text_slots.get(text)
        if slot is not None:
            return slot
        slot = len(self._runtime_text_handles)
        self._runtime_text_slots[text] = slot
        self._runtime_text_handles.append(address)
        return slot

    def _lower_runtime_text_handles(self, fn: _F) -> _F:
        """Move captured ``sim.log_text`` pointers into a runtime sidecar."""
        namespace = _closure_namespace(fn)
        try:
            node = copy.deepcopy(_function_def_from_source(fn))
        except (OSError, TypeError):
            return fn
        if not node.args.args:
            return fn
        env_name = node.args.args[0].arg
        local_names = {
            arg.arg
            for arg in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }
        local_names.update(
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name)
            and isinstance(child.ctx, (ast.Store, ast.Del))
        )
        lowerer = _RuntimeTextHandleLowerer(
            namespace=namespace,
            local_names=local_names,
            register=self._register_runtime_text_handle,
            env_name=env_name,
        )
        lowered = lowerer.visit(node)
        if not lowerer.changed:
            return fn
        if not isinstance(lowered, ast.FunctionDef):
            raise TypeError("runtime text lowering produced a non-function")
        lowered.decorator_list = []
        lowered.returns = None
        lowered.type_comment = None
        for arg in lowered.args.args:
            arg.annotation = None
            arg.type_comment = None
        namespace["_CIMBA_RUNTIME_TEXT_HANDLE"] = _runtime_text_handle
        return _compile_lowered(
            lowered,
            filename=(
                f"<cimba runtime text callback "
                f"'{self.name}.{fn.__name__}'>"
            ),
            fn_name=fn.__name__,
            qualname=fn.__qualname__,
            namespace=namespace,
            like=fn,
        )

    def _process_dag_blocks(
        self,
        entity_kinds: Mapping[str, str],
        extra_members: Mapping[str, Iterable[str]] | None = None,
    ) -> tuple[ProcessDAGBlock, ...]:
        process_names = {process.name for process in self._processes}
        extras = extra_members or {}
        return tuple(
            ProcessDAGBlock(
                decl.display_name or decl.name,
                tuple(dict.fromkeys((
                    *decl.dag_members(process_names, entity_kinds),
                    *extras.get(decl.name, ()),
                ))),
                kind=("component_collection" if decl.collection
                      else "component"),
            )
            for root in self._component_roots.values()
            for decl in root.walk()
        )

    # --- Internal callback registration ----------------------------------
    def _register_process(self, fn, *, copies: int = 1, priority: int = 0,
                struct=None, spawnable: bool = False,
                _spawn_field: str | None = None,
                _spawn_index: int | None = None,
                _process_field: str | None = None,
                _process_offset: int = 0,
                _owner_cls: type | None = None):
        """Register a process function `def fn(env)` or `def fn(env, idx)`
        (the latter receives its copy index). A final parameter annotated
        with a sim.Struct subclass receives the process's own field view:
        `def fn(env, vip: Visitor)` or `def fn(env, idx, vip: Visitor)`.
        copies=n starts n identical processes; priority sets the cimba
        process priority; struct= attaches the per-process fields without
        the view parameter. spawnable=True leaves a process unstarted at
        setup and publishes it as env.<name>, so sim.spawn(env.<name>, env)
        creates it at runtime. Component processes use the same decorator
        and publish their descriptor at the component path."""
        if getattr(fn, "__cimba_component_function__", False):
            raise ValueError(
                f"component function '{fn.__qualname__}' cannot be "
                "registered as a model process")
        if copies < 1:
            raise ValueError("copies must be >= 1")
        if struct is not None and not _is_struct_class(struct):
            raise ValueError("struct= expects a sim.Struct subclass")
        name = fn.__name__
        if _process_field is not None:
            if _spawn_field is not None:
                raise ValueError(
                    f"internal process binding for '{name}' cannot also bind "
                    "a Spawnable field")
            if _process_field not in self._process_fields:
                raise ValueError(
                    f"process binding for '{name}' references unknown "
                    f"Processes field '{_process_field}'")
            shape = self._field_shapes.get(_process_field)
            if shape is not None and len(shape) != 1:
                raise ValueError(
                    f"Processes field '{_process_field}' has unsupported "
                    f"shape {shape}")
            if shape is None and _process_offset != 0:
                raise ValueError(
                    f"scalar Processes field '{_process_field}' cannot use "
                    "a process offset")
            if shape is not None and (
                    _process_offset < 0
                    or _process_offset + copies > shape[0]):
                raise ValueError(
                    f"Processes field '{_process_field}' cannot hold process "
                    f"'{name}' at offset {_process_offset} with {copies} "
                    "copies")
        if _spawn_field is not None:
            if _spawn_field not in self._component_spawnable_fields:
                raise ValueError(
                    f"internal spawn binding for '{name}' references "
                    f"unknown component Spawnable field '{_spawn_field}'")
            shape = self._field_shapes.get(_spawn_field)
            if shape is None:
                if _spawn_index is not None:
                    raise ValueError(
                        f"Spawnable field '{_spawn_field}' is scalar but "
                        "got an indexed process binding")
            else:
                if len(shape) != 1:
                    raise ValueError(
                        f"Spawnable field '{_spawn_field}' has unsupported "
                        f"shape {shape}")
                if (_spawn_index is None or _spawn_index < 0
                        or _spawn_index >= shape[0]):
                    raise ValueError(
                        f"Spawnable field '{_spawn_field}' needs an index "
                        f"in [0, {shape[0]})")

        public_spawnable = (
            name in self._spawnable_fields
            and name not in self._component_spawnable_fields
        )
        generated_spawnable = spawnable and not public_spawnable \
            and _spawn_field is None
        spawnable = spawnable or public_spawnable or _spawn_field is not None
        spawn_field = _spawn_field if _spawn_field is not None else (
            name if spawnable else None)
        spawn_index = _spawn_index if _spawn_field is not None else None
        process_field = _process_field
        process_offset = _process_offset if _process_field is not None else 0
        publishes_field = (
            process_field is not None
            or public_spawnable
            or generated_spawnable
            or (spawn_field is not None and name == spawn_field)
        )
        if publishes_field:
            # The declared field publishes the handles (Processes) or the
            # spawn reference (Spawnable)
            if self._compiled is not None:
                raise RuntimeError("model is already compiled")
            if any(p.name == name for p in self._processes):
                raise ValueError(f"process '{name}' already registered")
            if ((process_field is not None and process_field != name)
                    or generated_spawnable):
                self._register_name(name, "process")
        else:
            self._register_name(name, "process")
        if spawn_field is not None:
            for p in self._processes:
                if (p.spawn_field == spawn_field
                        and p.spawn_index == spawn_index):
                    label = _spawnable_slot_label(spawn_field, spawn_index)
                    raise ValueError(
                        f"Spawnable field '{label}' already has a process "
                        "binding")

        nargs = fn.__code__.co_argcount
        params = fn.__code__.co_varnames[:nargs]
        localns = ({base.__name__: base for base in _owner_cls.__mro__}
                   if _owner_cls is not None else None)
        hints = get_type_hints(fn, localns=localns)
        own = hints.get(params[-1]) if nargs > 1 else None
        injected = _is_struct_class(own)
        for p in params[1:len(params) - 1 if injected else None]:
            if _is_struct_class(hints.get(p)):
                raise ValueError(f"process '{name}': the {hints[p].__name__}"
                                 " view must be the last parameter")
        if injected:
            if struct is not None and struct is not own:
                raise ValueError(f"process '{name}': struct= and the view "
                                 "annotation disagree")
            struct = own
        indexed = nargs - injected == 2
        if nargs - injected not in (1, 2):
            raise ValueError(
                "process functions take (env), (env, idx), and optionally "
                "a final view parameter annotated with a sim.Struct "
                "subclass")
        if spawnable:
            if copies != 1:
                raise ValueError(f"spawnable process '{name}' cannot take "
                                 "copies; sim.spawn() creates them")
            if indexed:
                raise ValueError(f"spawnable process '{name}' takes (env) "
                                 "or (env, view), not a copy index")
        fn = self._lower_component_refs(fn)
        fn = self._lower_dataset_methods(fn)
        fn = self._lower_history_methods(fn)
        fn = self._lower_entity_methods(fn)
        fn = self._lower_random_calls(fn)
        fn = self._lower_runtime_text_handles(fn)
        self._processes.append(_ProcDecl(name, fn, copies, priority,
                                         indexed, struct, injected,
                                         spawnable, spawn_field,
                                         spawn_index, process_field,
                                         process_offset))
        return fn

    def process_dag(self, *, validate: bool = True) -> ProcessDAG:
        """Infer a resource-aware graph from class-declared processes.

        ``validate`` is accepted for API stability. Inferred graphs may contain
        legitimate resource cycles, so acyclicity is checked only when callers
        explicitly ask for :meth:`ProcessDAG.topological_order`.
        """
        entity_kinds = {f.name: f.kind.name
                        for f in self._decls.fields.values()
                        if f.kind.dag_entity}
        # Registered events without a declared field publish their address
        # in a hidden _ev_<name> field.
        entity_kinds.update({field: "event"
                             for _n, _fn, field, _d in self._events})
        spawnable_field_processes: dict[str, list[str]] = {}
        spawnable_index_processes: dict[tuple[str, int], list[str]] = {}
        process_field_processes: dict[str, list[str]] = {}
        process_index_processes: dict[tuple[str, int], list[str]] = {}
        for process in self._processes:
            if process.spawnable and process.spawn_field is not None:
                spawnable_field_processes.setdefault(
                    process.spawn_field, []).append(process.name)
                if process.spawn_index is not None:
                    spawnable_index_processes.setdefault(
                        (process.spawn_field, process.spawn_index),
                        [],
                    ).append(process.name)
            if not process.spawnable and process.process_field is not None:
                process_field_processes.setdefault(
                    process.process_field, []).append(process.name)
                for slot in range(process.process_offset,
                                  process.process_offset + process.copies):
                    process_index_processes.setdefault(
                        (process.process_field, slot),
                        [],
                    ).append(process.name)

        function_nodes: dict[str, ProcessDAGNode] = {}
        function_edges: list[ProcessDAGEdge] = []
        function_members: dict[str, list[str]] = {}

        def add_function_member(decl_name: str, key: str) -> None:
            members = function_members.setdefault(decl_name, [])
            if key not in members:
                members.append(key)

        for spec in self._component_functions.values():
            function_node = ProcessDAGNode(spec.graph_name, "function")
            function_nodes[function_node.key] = function_node
            add_function_member(spec.decl.name, function_node.key)
            for dependency in spec.dependencies:
                if not dependency.direct:
                    continue
                access = dependency.access
                if access.field in access.decl.constants:
                    continue
                field_kind = access.decl.decls.kind_of(access.field)
                if field_kind not in ("param", "output", "state", "fstate"):
                    continue
                flat_name = access.decl.direct_field_map[access.field]
                field_node = ProcessDAGNode(flat_name, field_kind)
                function_nodes[field_node.key] = field_node
                add_function_member(access.decl.name, field_node.key)
                edge = ProcessDAGEdge(
                    field_node.key, function_node.key, "read")
                if edge not in function_edges:
                    function_edges.append(edge)
            for callee in spec.callees:
                edge = ProcessDAGEdge(
                    function_node.key, f"function:{callee}", "call")
                if edge not in function_edges:
                    function_edges.append(edge)

        for process in self._processes:
            for called in getattr(
                    process.fn, "__cimba_function_calls__", ()):
                edge = ProcessDAGEdge(
                    f"process:{process.name}",
                    f"function:{called}",
                    "call",
                )
                if edge not in function_edges:
                    function_edges.append(edge)
        return infer_process_dag(
            self._processes,
            entity_kinds=entity_kinds,
            process_fields=self._process_fields,
            spawnable_fields=self._spawnable_fields,
            spawnable_field_processes=spawnable_field_processes,
            spawnable_index_processes=spawnable_index_processes,
            process_field_processes=process_field_processes,
            process_index_processes=process_index_processes,
            event_callbacks=((field, fn) for _n, fn, field, _d in self._events),
            blocks=self._process_dag_blocks(
                entity_kinds, function_members),
            extra_nodes=function_nodes.values(),
            extra_edges=function_edges,
        )

    def _register_predicate(
        self, fn: _F, *, target_field: str | None = None,
    ) -> _F:
        """Register a condition predicate `def fn(env) -> bool`. Its
        compiled address is published in the declared Predicate field of
        the same name, for use with env.<cond>.wait_for(env.<name>).
        (Without a declared field, it is published as the hidden
        field `_pred_<name>`.)"""
        name = fn.__name__
        if fn.__code__.co_argcount != 1:
            raise ValueError("predicate functions take (env)")
        localns = {base.__name__: base for base in type(self).__mro__}
        if get_type_hints(fn, localns=localns).get("return") is not bool:
            raise ValueError("predicate functions must return bool")
        if target_field is not None:
            if target_field not in self._predicate_fields:
                raise ValueError(
                    f"predicate callback '{name}' field= must name a "
                    "declared sim.Predicate field")
            if self._compiled is not None:
                raise RuntimeError("model is already compiled")
            if any(f == target_field for _n, _fn, f in self._predicates):
                raise ValueError(
                    f"predicate field '{target_field}' already bound")
            self._register_name(name, "predicate")
            field = target_field
        else:
            self._register_name(name, "predicate")
            field = f"_pred_{name}"
        fn = self._lower_component_refs(fn)
        fn = self._lower_dataset_methods(fn)
        fn = self._lower_history_methods(fn)
        fn = self._lower_entity_methods(fn)
        fn = self._lower_random_calls(fn)
        fn = self._lower_runtime_text_handles(fn)
        self._predicates.append((name, fn, field))
        return fn

    def _register_event(
        self, fn: _F, *, target_field: str | None = None,
    ) -> _F:
        """Register a low-level event callback `def fn(env)` or
        `def fn(env, data)` (the latter receives the int64 data word given
        at scheduling time). Its compiled address is published in the
        declared Event field of the same name, for use with
        env.<name>.schedule(delay, ...). (Without a declared field, it is
        published as the hidden field `_ev_<name>`.)"""
        name = fn.__name__
        nargs = fn.__code__.co_argcount
        if nargs not in (1, 2):
            raise ValueError("event functions take (env) or (env, data)")
        if target_field is not None:
            if target_field not in self._event_fields:
                raise ValueError(
                    f"event callback '{name}' field= must name a declared "
                    "sim.Event field")
            if self._compiled is not None:
                raise RuntimeError("model is already compiled")
            if any(f == target_field for _n, _fn, f, _d in self._events):
                raise ValueError(f"event field '{target_field}' already bound")
            self._register_name(name, "event")
            field = target_field
        else:
            self._register_name(name, "event")
            field = f"_ev_{name}"
        self.entity_fields[field] = "event"
        fn = self._lower_component_refs(fn)
        fn = self._lower_dataset_methods(fn)
        fn = self._lower_history_methods(fn)
        fn = self._lower_entity_methods(fn, extra_fields={field: "event"})
        fn = self._lower_random_calls(fn)
        fn = self._lower_runtime_text_handles(fn)
        self._events.append((name, fn, field, nargs == 2))
        return fn

    def _register_collect(self, fn: _F) -> _F:
        """Register the statistics-collection function, run once at the
        end of each trial, after any component-owned @sim.collect methods
        (so it can aggregate over component outputs)."""
        if self._collect is not None:
            raise ValueError("collect() already registered")
        if self._compiled is not None:
            raise RuntimeError("model is already compiled")
        if fn.__code__.co_argcount != 1:
            raise ValueError("model collect functions take (env)")
        fn = self._lower_component_refs(fn)
        fn = self._lower_history_capture_methods(fn)
        fn = self._lower_dataset_capture_methods(fn)
        fn = self._lower_dataset_methods(fn)
        fn = self._lower_history_methods(fn)
        fn = self._lower_entity_methods(fn)
        fn = self._lower_random_calls(fn)
        fn = self._lower_runtime_text_handles(fn)
        self._collect = fn
        return fn

    @property
    def _collects(self) -> list[tuple[Callable[..., Any], int]]:
        """All end-of-trial collect functions in execution order --
        component-owned collects first, the model-level one last -- each
        with the number of instances it is called for (multi-instance
        collects take the instance index as their second argument)."""
        fns = list(self._component_collects)
        if self._collect is not None:
            fns.append((self._collect, 1))
        return fns

    def _register_name(self, name: str, kind: str) -> None:
        _check_name(name, kind)
        if name in self._seen:
            raise ValueError(f"duplicate name '{name}'")
        if self._compiled is not None:
            raise RuntimeError("model is already compiled")
        self._seen.add(name)

    # --- Trial record layout ----------------------------------------------
    @property
    def _entities(self) -> list[str]:
        return self._decls.names("queue", "resource", "pool", "store",
                                 "dataset", "condition")

    def _runtime_process_descriptors(
        self,
        dtype: np.dtype,
        callbacks: Mapping[str, Any],
        native_names: Mapping[str, str],
    ) -> tuple[np.ndarray, int]:
        """Build the fixed-width process table consumed by lifecycle ABI."""
        rows: list[list[int]] = []
        handle_start = 0
        fields = dtype.fields or {}
        for process in self._processes:
            if process.spawnable:
                continue
            destination = -1
            if process.process_field is not None:
                destination = (
                    fields[process.process_field][1]
                    + 8 * process.process_offset
                )
            rows.append([
                callbacks[process.name].address,
                _b.cstring(native_names[process.name]),
                process.alloc_size,
                process.priority,
                process.copies,
                int(process.indexed),
                handle_start,
                destination,
            ])
            handle_start += process.copies
        descriptors = np.asarray(rows, dtype=np.int64)
        if not rows:
            descriptors = np.zeros(
                (1, _PROCESS_DESCRIPTOR_WIDTH), dtype=np.int64)
        return descriptors, handle_start

    def _runtime_entity_descriptors(self, dtype: np.dtype) -> np.ndarray:
        """Build entity setup/recording/cleanup descriptors for one model."""
        fields = dtype.fields or {}
        kind_codes = {
            "queue": _ENTITY_BUFFER,
            "resource": _ENTITY_RESOURCE,
            "pool": _ENTITY_RESOURCEPOOL,
            "store": _ENTITY_OBJECTQUEUE,
            "dataset": _ENTITY_DATASET,
            "condition": _ENTITY_CONDITION,
        }
        logical_names = [
            native_name
            for field in self._decls.by_kind(
                "queue", "resource", "pool", "store", "condition")
            for _key, native_name in self._field_name_keys(field.name)
        ]
        logical_names.extend(
            f"{field}_{index}"
            for field, count in self.pqueues.items()
            for index in range(count)
        )
        native_names = _native_names(logical_names)
        rows: list[list[int]] = []
        for field in self._decls.by_kind(
                "queue", "resource", "pool", "store", "dataset",
                "condition"):
            names = self._field_name_keys(field.name)
            for index, (_key, native_name) in enumerate(names):
                capacity_mode = _CAPACITY_CONSTANT
                capacity = -1
                if isinstance(field.capacity, int):
                    capacity = field.capacity
                elif isinstance(field.capacity, str):
                    capacity_mode = _CAPACITY_FIELD
                    slot = index
                    if field.capacity_slots is not None:
                        slot = field.capacity_slots[index]
                    capacity = fields[field.capacity][1] + 8 * slot
                rows.append([
                    kind_codes[field.kind.name],
                    (0 if field.kind.name == "dataset"
                     else _b.cstring(native_names[native_name])),
                    capacity_mode,
                    capacity,
                    fields[field.name][1] + 8 * index,
                ])
        for field, count in self.pqueues.items():
            offset = fields[field][1]
            for index in range(count):
                rows.append([
                    _ENTITY_PRIORITYQUEUE,
                    _b.cstring(native_names[f"{field}_{index}"]),
                    _CAPACITY_CONSTANT,
                    -1,
                    offset + 8 * index,
                ])
        descriptors = np.asarray(rows, dtype=np.int64)
        if not rows:
            descriptors = np.zeros(
                (1, _ENTITY_DESCRIPTOR_WIDTH), dtype=np.int64)
        return descriptors

    def _field_spec(self, name: str, fmt: str) -> tuple[Any, ...]:
        shape = self._field_shapes.get(name)
        if shape is None:
            return (name, fmt)
        return (name, fmt, shape)

    def _param_axes(self, param_values: Mapping[str, Any]) -> list[np.ndarray]:
        return [
            _as_param_axis(param_values[p], self._field_shapes.get(p), p)
            for p in self.params
        ]

    def _resolve_param_values(
        self,
        param_values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Fill omitted Params from declared defaults and validate names."""
        resolved = dict(param_values)
        schemas = {
            schema.flattened_name: schema
            for schema in self.component_schema()
            if schema.kind == "param"
        }
        for name, value in tuple(resolved.items()):
            schema = schemas.get(name)
            if schema is not None and isinstance(value, Mapping):
                resolved[name] = self._resolve_indexed_component_param(
                    name, value, schema)
        for name, default in self.param_defaults.items():
            resolved.setdefault(name, default)
        missing = set(self.params) - set(resolved)
        if missing:
            raise ValueError(f"missing parameter values: {sorted(missing)}")
        unknown = set(param_values) - set(self.params) - set(self.traces)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        return resolved

    def _resolve_indexed_component_param(
        self,
        name: str,
        value: Mapping[Any, Any],
        schema: ComponentFieldSchema,
    ) -> Any:
        """Convert ``{logical component index: value}`` to packed order."""
        invalid = [key for key in value if type(key) is not int]
        if invalid:
            raise TypeError(
                f"component parameter '{name}' owner indexes must be ints")
        owner_set = set(schema.owners)
        extra = set(value) - owner_set
        if extra:
            raise ValueError(
                f"component parameter '{name}' has non-owning indexes: "
                f"{sorted(extra)}")

        defaults = self.param_defaults.get(name, _MISSING)
        if defaults is _MISSING:
            default_by_owner: dict[int, Any] = {}
        else:
            packed_defaults = (
                (defaults,) if schema.shape is None else tuple(defaults))
            default_by_owner = dict(zip(schema.owners, packed_defaults))
        missing = owner_set - set(value) - set(default_by_owner)
        if missing:
            raise ValueError(
                f"component parameter '{name}' is missing owner indexes: "
                f"{sorted(missing)}")
        ordered = [
            value[owner] if owner in value else default_by_owner[owner]
            for owner in schema.owners
        ]
        if schema.shape is None:
            return ordered[0]
        arrays = [np.asarray(item, dtype=np.float64) for item in ordered]
        if all(item.ndim == 0 for item in arrays):
            return np.asarray(ordered, dtype=np.float64)
        if (all(item.ndim == 1 for item in arrays)
                and len({item.shape for item in arrays}) == 1):
            return np.column_stack(arrays)
        raise ValueError(
            f"component parameter '{name}' indexed values must be all "
            "scalars or equal-length 1-D sweep arrays")

    def _trace_field_spec(self, name: str) -> tuple[Any, ...]:
        shape = self._field_shapes.get(name)
        if shape is None:
            return (name, "<i8", (2,))
        if len(shape) != 1:
            raise ValueError(f"trace field '{name}' has unsupported "
                             f"shape {shape}")
        return (name, "<i8", (*shape, 2))

    def _field_name_keys(self, name: str) -> list[tuple[str, str]]:
        shape = self._field_shapes.get(name)
        if shape is None:
            return [(f"NAME_{name}", name)]
        if len(shape) != 1:
            raise ValueError(f"field '{name}' has unsupported shape {shape}")
        return [(f"NAME_{name}_{i}", f"{name}_{i}")
                for i in range(shape[0])]

    @property
    def dtype(self) -> np.dtype:
        # (name, format) or (name, format, shape) numpy field specs
        fields: list[Any] = list(_LIFECYCLE_ABI_FIELDS)
        for f in self._decls.by_kind("param", "output", "queue", "resource",
                                     "pool", "store", "dataset", "condition",
                                     "state", "fstate"):
            fields.append((f.name, f.kind.fmt) if f.shape is None
                          else (f.name, f.kind.fmt, f.shape))
        fields += [self._trace_field_spec(t) for t in self.traces]
        fields += [(f.name, "<i8", (f.count,))
                   for f in self._decls.by_kind("pqueues")]
        fields += [(p, "<i8") for p in self._predicate_fields]
        fields += [(f, "<i8") for _n, _fn, f in self._predicates
                   if f.startswith("_pred_")]
        fields += [(e, "<i8") for e in self._event_fields]
        fields += [(f, "<i8") for _n, _fn, f, _d in self._events
                   if f.startswith("_ev_")]
        fields += [self._field_spec(s, "<i8") for s in self._spawnable_fields]
        fields += [(p.spawn_field, "<i8") for p in self._processes
                   if p.spawnable and p.spawn_field is not None
                   and p.spawn_field not in self._spawnable_fields
                   and p.spawn_field not in self._component_spawnable_fields]
        if self._history_captures or self._dataset_captures:
            fields += [
                (HISTORY_CAPTURE_TRIAL_FIELD, "<u8"),
                (HISTORY_CAPTURE_STORE_FIELD, "<i8"),
            ]
        process_fields_added: set[str] = set()
        for p in self._processes:
            if p.spawnable:
                continue
            if p.process_field is not None:
                if p.process_field not in process_fields_added:
                    shape = self._field_shapes.get(p.process_field)
                    if shape is None:
                        fields += [(p.process_field, "<i8", (p.copies,))]
                    else:
                        fields += [self._field_spec(p.process_field, "<i8")]
                    process_fields_added.add(p.process_field)
        return np.dtype(fields)

    # --- Compilation --------------------------------------------------------
    def _compile_callbacks(
        self,
        rec: Any,
        extra_jobs: Sequence[tuple[Any, Callable[..., Any]]] = (),
        precompiled_procs: Mapping[str, Any] | None = None,
        precompiled_predicates: Mapping[str, Any] | None = None,
        precompiled_events: Mapping[str, Any] | None = None,
        precompiled_extra: Mapping[int, Any] | None = None,
        warm_parent: bool | None = None,
        cache_counters: _CacheCounters | None = None,
        processes: Sequence[_ProcDecl] | None = None,
        predicates: Sequence[tuple[str, Callable[..., Any], str]] | None = None,
        events: Sequence[tuple[str, Callable[..., Any], str, bool]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[Any]]:
        """Compile lowered class callbacks to Cimba's native callback ABIs."""
        trial_ptr = types.CPointer(rec)
        proc_sig = types.intp(types.intp, trial_ptr)
        proc_sig_ix = types.intp(types.intp, types.CPointer(types.int64))
        pred_sig = types.boolean(types.intp, types.intp, trial_ptr)
        # cmb_event_func: subject is the env pointer, object the data word
        ev_sig = types.void(trial_ptr, types.intp)
        rec_from_addr = ptr_caster(rec)

        def closure_namespace(fn):
            namespace = dict(fn.__globals__)
            if fn.__closure__ is not None:
                for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                    namespace[name] = cell.cell_contents
            return namespace

        def make_pred(inner):
            def pred(cnd, prc, ctxp):
                return inner(carray(ctxp, 1)[0])
            return pred

        def make_event(inner, takes_data):
            if takes_data:
                def ev(subject, data):
                    inner(carray(subject, 1)[0], data)
            else:
                def ev(subject, data):
                    inner(carray(subject, 1)[0])
            return ev

        def make_proc_indexed(p):
            """Build the indexed-process adapter; workers compile its body."""
            inner = njit(p.fn)
            struct = p.struct if p.injected else None
            if struct is None:
                def proc(me, ctxp):
                    pair = carray(ctxp, 2)
                    env = carray(rec_from_addr(pair[0]), 1)[0]
                    inner(env, pair[1])
                    return 0
            else:
                def proc(me, ctxp):
                    pair = carray(ctxp, 2)
                    env = carray(rec_from_addr(pair[0]), 1)[0]
                    inner(env, pair[1], struct(me))
                    return 0
            # The public indexed-process ABI carries an int64 context pointer,
            # not the model record type. The adapter nevertheless embeds a
            # record-specific pointer cast, so its cache identity must include
            # the complete record layout explicitly.
            proc.__cimba_cache_salt__ = (
                "indexed-record-v1",
                rec.dtype.descr,
                rec.dtype.itemsize,
            )
            return proc

        def make_proc_direct(p):
            """Build a non-indexed process body with the native callback ABI.

            The previous path compiled the lowered process with ``njit`` and
            then compiled a second callback whose only job was to unpack the
            record pointer and call that dispatcher.  Non-indexed processes
            can use the callback's record pointer directly, avoiding that
            duplicate Numba compilation while retaining the same ABI.
            """
            source = getattr(p.fn, "__cimba_source__", None)
            if source is None:
                source = inspect.getsource(p.fn)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
            fn_node = next(
                node for node in tree.body
                if isinstance(node, ast.FunctionDef)
            )
            env_name = fn_node.args.args[0].arg
            body = copy.deepcopy(fn_node.body)
            prefix = [
                ast.Assign(
                    targets=[ast.Name(id=env_name, ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Call(
                            func=ast.Name(id="carray", ctx=ast.Load()),
                            args=[ast.Name(id="ctxp", ctx=ast.Load()),
                                  ast.Constant(value=1)],
                            keywords=[],
                        ),
                        slice=ast.Constant(value=0),
                        ctx=ast.Load(),
                    ),
                ),
            ]
            namespace = closure_namespace(p.fn)
            namespace["carray"] = carray
            if p.struct is not None:
                view_name = fn_node.args.args[-1].arg
                prefix.append(
                    ast.Assign(
                        targets=[ast.Name(id=view_name, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(
                                id="_CIMBA_STRUCT_VIEW", ctx=ast.Load()),
                            args=[ast.Name(id="me", ctx=ast.Load())],
                            keywords=[],
                        ),
                    )
                )
                namespace["_CIMBA_STRUCT_VIEW"] = p.struct
            wrapper = ast.FunctionDef(
                name=f"_cimba_direct_{p.name}",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="me"), ast.arg(arg="ctxp")],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=prefix + body + [
                    ast.Return(value=ast.Constant(value=0))
                ],
                decorator_list=[],
            )
            ast.fix_missing_locations(wrapper)
            module = ast.Module(body=[wrapper], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(
                compile(module, f"<cimba direct process '{p.name}'>", "exec"),
                namespace,
            )
            return namespace[wrapper.name]

        def raise_process_compile_error(p, exc):
            calls = getattr(p.fn, "__cimba_function_calls__", ())
            if calls:
                raise TypeError(
                    f"process '{p.name}' has an invalid call to "
                    "component function(s): " + ", ".join(calls)
                ) from exc
            raise exc

        proc_cfuncs = dict(precompiled_procs or {})
        process_jobs = []
        compile_processes = self._processes if processes is None else processes
        for p in compile_processes:
            if p.name in proc_cfuncs:
                continue
            try:
                if p.indexed:
                    process_jobs.append((p, proc_sig_ix, make_proc_indexed(p)))
                else:
                    process_jobs.append((p, proc_sig, make_proc_direct(p)))
            except Exception as exc:
                raise_process_compile_error(p, exc)
        pending_extra = [
            (index, signature, function)
            for index, (signature, function) in enumerate(extra_jobs)
            if index not in (precompiled_extra or {})
        ]
        compile_predicates = self._predicates if predicates is None \
            else predicates
        compile_events = self._events if events is None else events
        predicate_jobs = [
            (name, field, pred_sig, make_pred(njit(fn)))
            for name, fn, field in compile_predicates
            if field not in (precompiled_predicates or {})
        ]
        event_jobs = [
            (name, field, ev_sig, make_event(njit(fn), takes_data))
            for name, fn, field, takes_data in compile_events
            if field not in (precompiled_events or {})
        ]
        all_jobs = [
            *((signature, function)
              for _p, signature, function in process_jobs),
            *((signature, function)
              for _name, _field, signature, function in predicate_jobs),
            *((signature, function)
              for _name, _field, signature, function in event_jobs),
            *((signature, function)
              for _index, signature, function in pending_extra),
        ]

        try:
            callbacks = _compile_cfuncs(
                all_jobs,
                warm_parent=(len(all_jobs) > 4
                             if warm_parent is None else warm_parent),
                cache_counters=cache_counters,
            )
        except Exception:
            # Recompile in the parent so invalid user code retains its full
            # process-specific diagnostic instead of a worker traceback.
            callbacks = []
            for p, signature, function in process_jobs:
                try:
                    callbacks.append(
                        _native_cfunc(signature)(function))
                except Exception as exc:
                    raise_process_compile_error(p, exc)
            for name, _field, signature, function in (
                    *predicate_jobs, *event_jobs):
                try:
                    callbacks.append(
                        _compile_parallel_cfunc(signature, function))
                except Exception as exc:
                    raise TypeError(
                        f"callback '{name}' failed to compile") from exc
            for _index, signature, function in pending_extra:
                callbacks.append(
                    _compile_parallel_cfunc(signature, function))
        for (p, _signature, _function), callback in zip(
                process_jobs, callbacks):
            proc_cfuncs[p.name] = callback

        # Predicates and events keyed by the env field that publishes
        # their compiled address
        offset = len(process_jobs)
        pred_cfuncs = dict(precompiled_predicates or {})
        pred_cfuncs.update({
            field: callback
            for (_name, field, _signature, _function), callback in zip(
                predicate_jobs, callbacks[offset:])
        })
        offset += len(predicate_jobs)
        event_cfuncs = dict(precompiled_events or {})
        event_cfuncs.update({
            field: callback
            for (_name, field, _signature, _function), callback in zip(
                event_jobs, callbacks[offset:])
        })
        offset += len(event_jobs)
        extra_callbacks = [None] * len(extra_jobs)
        for index, callback in (precompiled_extra or {}).items():
            extra_callbacks[index] = callback
        for (index, _signature, _function), callback in zip(
                pending_extra, callbacks[offset:]):
            extra_callbacks[index] = callback
        return proc_cfuncs, pred_cfuncs, event_cfuncs, extra_callbacks

    @staticmethod
    def _direct_collect_callback(fn, index: int, count: int):
        """Build a collect body with the native callback ABI.

        Multi-instance component collectors execute their lowered body in one
        native callback loop, avoiding a separate lazy ``njit`` dispatcher and
        repeated generated calls from trial teardown.
        """
        source = getattr(fn, "__cimba_source__", None)
        if source is None:
            source = inspect.getsource(fn)
        tree = ast.parse(textwrap.dedent(source))
        fn_node = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
        )
        env_name = fn_node.args.args[0].arg
        namespace = dict(fn.__globals__)
        if fn.__closure__ is not None:
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                namespace[name] = cell.cell_contents
        namespace["carray"] = carray
        body: list[ast.stmt] = [
            ast.Assign(
                targets=[ast.Name(id=env_name, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Call(
                        func=ast.Name(id="carray", ctx=ast.Load()),
                        args=[ast.Name(id="ctxp", ctx=ast.Load()),
                              ast.Constant(value=1)],
                        keywords=[],
                    ),
                    slice=ast.Constant(value=0),
                    ctx=ast.Load(),
                ),
            ),
        ]
        if count == 1:
            body.extend(copy.deepcopy(fn_node.body))
        else:
            if len(fn_node.args.args) < 2:
                raise TypeError(
                    "multi-instance collect callback needs an index argument"
                )
            index_name = fn_node.args.args[1].arg
            loop_index = f"_cimba_collect_index_{index}"
            body.append(ast.For(
                target=ast.Name(id=loop_index, ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[ast.Constant(value=count)],
                    keywords=[],
                ),
                body=[
                    ast.Assign(
                        targets=[ast.Name(id=index_name, ctx=ast.Store())],
                        value=ast.Name(id=loop_index, ctx=ast.Load()),
                    ),
                    *copy.deepcopy(fn_node.body),
                ],
                orelse=[],
            ))
        wrapper = ast.FunctionDef(
            name=f"_cimba_direct_collect_{index}",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="ctxp")],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[],
        )
        module = ast.Module(body=[wrapper], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(
            compile(module, f"<cimba direct collect {index}>", "exec"),
            namespace,
        )
        return namespace[wrapper.name]

    def _aot_lifecycle_key(self) -> tuple[str, ...]:
        """Version key for the layout-independent lifecycle callback ABI."""
        return ("cimba-lifecycle-abi-v2",)

    def _compile(self) -> _Compiled:
        if self._compiled is not None:
            return self._compiled
        if type(self).__cimba_precompile__ == "lazy":
            self._ensure_class_precompiled()
        if not self._processes:
            raise ValueError("model has no processes")
        bound = {f for _n, _fn, f in self._predicates}
        unbound = [f for f in self._predicate_fields if f not in bound]
        if unbound:
            raise ValueError(f"Predicate field(s) {unbound} declared but "
                             "no @predicate of that name registered")
        bound = {f for _n, _fn, f, _d in self._events}
        unbound = [f for f in self._event_fields if f not in bound]
        if unbound:
            raise ValueError(f"Event field(s) {unbound} declared but "
                             "no @event of that name registered")
        process_field_slots: dict[str, dict[int, int]] = {}
        for process in self._processes:
            if process.spawnable or process.process_field is None:
                continue
            slots = process_field_slots.setdefault(process.process_field, {})
            for slot in range(process.process_offset,
                              process.process_offset + process.copies):
                slots[slot] = slots.get(slot, 0) + 1
        unbound_process_fields: list[str] = []
        bad_process_fields: list[str] = []
        for field in self._process_fields:
            slots = process_field_slots.get(field)
            if slots is None:
                unbound_process_fields.append(field)
                continue
            shape = self._field_shapes.get(field)
            if shape is None:
                expected = len(slots)
            elif len(shape) == 1:
                expected = shape[0]
            else:
                raise ValueError(f"Processes field '{field}' has unsupported "
                                 f"shape {shape}")
            if (set(slots) != set(range(expected))
                    or any(count != 1 for count in slots.values())):
                bad_process_fields.append(field)
        if unbound_process_fields:
            raise ValueError(
                f"Processes field(s) {unbound_process_fields} declared but "
                "no @process of that name registered")
        if bad_process_fields:
            raise ValueError(
                f"Processes field(s) {bad_process_fields} have incomplete or "
                "overlapping process handle bindings")
        bound_spawn_slots = {
            (p.spawn_field, p.spawn_index)
            for p in self._processes
            if p.spawnable
        }
        expected_spawn_slots: list[tuple[str, int | None]] = []
        for field in self._spawnable_fields:
            shape = self._field_shapes.get(field)
            if shape is None:
                expected_spawn_slots.append((field, None))
            elif len(shape) == 1:
                expected_spawn_slots.extend(
                    (field, index) for index in range(shape[0]))
            else:
                raise ValueError(f"Spawnable field '{field}' has "
                                 f"unsupported shape {shape}")
        unbound = [
            _spawnable_slot_label(field, index)
            for field, index in expected_spawn_slots
            if (field, index) not in bound_spawn_slots
        ]
        if unbound:
            raise ValueError(f"Spawnable field(s) {unbound} declared but "
                             "no @process of that name registered")

        dtype = self.dtype
        rec = from_dtype(dtype)
        trial_ptr = types.CPointer(rec)

        direct_collects = [
            (index, self._direct_collect_callback(fn, index, count))
            for index, (fn, count) in enumerate(self._collects)
        ]
        native_process_names = _native_names(
            process.name for process in self._processes)
        self._source = "# fixed cimba lifecycle ABI v2"

        aot_procs, aot_preds, aot_events, precompiled_extra = \
            self._aot_class_callbacks()
        cache_counters = _CacheCounters()
        proc_cfuncs, pred_cfuncs, event_cfuncs, lifecycle = \
            self._compile_callbacks(
                rec,
                [
                    (types.void(_LIFECYCLE_ABI_PTR, types.intp),
                     _runtime_recording_event),
                    (types.void(_LIFECYCLE_ABI_PTR),
                     _runtime_trial_initialize),
                    (types.void(_LIFECYCLE_ABI_PTR),
                     _runtime_trial_entities),
                    (types.void(_LIFECYCLE_ABI_PTR),
                     _runtime_trial_processes),
                    (types.void(_LIFECYCLE_ABI_PTR),
                     _runtime_trial_teardown),
                    (types.void(_LIFECYCLE_ABI_PTR), _runtime_trial),
                    (types.void(_LIFECYCLE_ABI_PTR), _runtime_trial_stop),
                    (types.void(_LIFECYCLE_ABI_PTR),
                     _runtime_trial_process_cleanup),
                    (types.void(_LIFECYCLE_ABI_PTR),
                     _runtime_trial_collect),
                    *((types.void(trial_ptr), function)
                      for _index, function in direct_collects),
                ],
                aot_procs,
                aot_preds,
                aot_events,
                precompiled_extra,
                cache_counters=cache_counters,
            )
        self._callback_cache_stats = CompilationCacheStats(
            hits=cache_counters.hits,
            misses=cache_counters.misses,
            writes=cache_counters.writes,
        )
        (recording_event, trial_initialize, trial_entities, trial_processes,
         trial_teardown, trial, trial_stop, trial_process_cleanup,
         trial_collect, *collect_callbacks) = lifecycle
        compiled_collects = {
            index: callback
            for (index, _function), callback in zip(
                direct_collects, collect_callbacks)
        }
        spawn_descs = {
            p.name: np.array([proc_cfuncs[p.name].address,
                              _b.cstring(native_process_names[p.name]),
                              p.alloc_size],
                             dtype=np.int64)
            for p in self._processes if p.spawnable
        }
        spawn_assignments = tuple(
            (p.spawn_field, p.spawn_index, p.name)
            for p in self._processes
            if p.spawnable and p.spawn_field is not None
        )
        process_descriptors, process_handle_count = \
            self._runtime_process_descriptors(
                dtype, proc_cfuncs, native_process_names)
        entity_descriptors = self._runtime_entity_descriptors(dtype)
        entity_descriptor_count = (
            sum(
                1 if field.shape is None else field.shape[0]
                for field in self._decls.by_kind(
                    "queue", "resource", "pool", "store", "dataset",
                    "condition")
            )
            + sum(self.pqueues.values())
        )
        collect_descriptors = np.asarray(
            [compiled_collects[index].address
             for index in range(len(compiled_collects))],
            dtype=np.int64,
        )
        if collect_descriptors.size == 0:
            collect_descriptors = np.zeros(1, dtype=np.int64)
        runtime_text_handles = np.asarray(
            self._runtime_text_handles, dtype=np.int64)
        if runtime_text_handles.size == 0:
            runtime_text_handles = np.zeros(1, dtype=np.int64)

        # Keep every compiled artifact alive for the model's lifetime
        self._compiled = {
            "trial": trial,
            "events": (
                recording_event,
                trial_initialize,
                trial_entities,
                trial_processes,
                trial_teardown,
                trial_stop,
                trial_process_cleanup,
                trial_collect,
            ),
            "procs": proc_cfuncs,
            "preds": pred_cfuncs,
            "user_events": event_cfuncs,
            "collect_callbacks": compiled_collects,
            "collect_descriptors": collect_descriptors,
            "process_descriptors": process_descriptors,
            "process_handle_count": process_handle_count,
            "entity_descriptors": entity_descriptors,
            "entity_descriptor_count": entity_descriptor_count,
            "runtime_text_handles": runtime_text_handles,
            "spawns": spawn_descs,
            "spawn_assignments": spawn_assignments,
            "dtype": dtype,
        }
        return self._compiled

    # --- Experiments ----------------------------------------------------------
    def trial_seeds(self, *,
                    seed: int,
                    replications: int = 1,
                    **param_values: Any) -> np.ndarray:
        """The per-trial seeds that experiment() with this seed, these
        swept parameter values, and this replication count will assign,
        in trial order (design-point-major, replications innermost).

        Use this to generate trace data outside experiment() -- e.g. in
        parallel when a generator is expensive -- while staying
        reproducible from the experiment seed: feed
        ``trace_rng(seeds[i], field_name)`` to the generator and pass
        the finished rows to experiment() with the same seed. Trace
        fields passed here are ignored, so the experiment() keyword
        arguments can be reused as-is."""
        param_values = self._resolve_param_values(param_values)
        if replications < 1:
            raise ValueError("replications must be >= 1")
        n_points = _n_design_points(self._param_axes(param_values))
        return _draw_trial_seeds(seed, n_points * replications)

    def experiment(self,
                   *,
                   replications: int = 1,
                   duration: float = 1.0e6,
                   warmup: float = 1.0e3,
                   cooldown: float = 0.0,
                   start_time: float = 0.0,
                   seed: int | None = None,
                   **param_values: "ArrayLike | Callable[..., ArrayLike]",
                   ) -> "Experiment[_ExperimentResultT]":
        """Build an experiment: the cross product of the swept parameter
        values (scalars are held fixed), replicated with distinct seeds.
        Omitted Params use their declaration defaults; Params without a
        default remain required.

        Trace fields take their replay data here as well: a 1-D array
        shared by every trial, a 2-D array whose row i replays in trial i
        (trial order is design-point-major with replications innermost),
        or a sequence of 1-D arrays for ragged per-trial traces.

        A trace field also accepts a callable ``f(rng)`` or
        ``f(rng, trial_index)`` returning a 1-D array; it is invoked once
        per trial with ``trace_rng(trial_seed, field_name)``, a numpy
        Generator derived from that trial's own seed, so the experiment
        ``seed`` reproduces the generated traces too. A callable's
        ``trace_rng_name`` attribute overrides the field name in that
        derivation (see ``trace_rng``)."""
        compiled = self._compile()

        param_values = self._resolve_param_values(param_values)
        missing_traces = set(self.traces) - set(param_values)
        if missing_traces:
            raise ValueError(f"missing trace values: "
                             f"{sorted(missing_traces)}")
        if replications < 1:
            raise ValueError("replications must be >= 1")

        axes = self._param_axes(param_values)
        n_points = _n_design_points(axes)
        n_trials = n_points * replications

        trials = np.zeros(n_trials, dtype=compiled["dtype"])
        trials["start_time"] = start_time
        trials["warmup_s"] = warmup
        trials["duration_s"] = duration
        trials["cooldown_s"] = cooldown
        trials[_RECORDING_EVENT_FIELD] = compiled["events"][0].address
        trials[_TRIAL_INITIALIZE_FIELD] = compiled["events"][1].address
        trials[_TRIAL_ENTITIES_FIELD] = compiled["events"][2].address
        trials[_TRIAL_PROCESSES_FIELD] = compiled["events"][3].address
        trials[_TRIAL_TEARDOWN_FIELD] = compiled["events"][4].address
        trials[_TRIAL_STOP_FIELD] = compiled["events"][5].address
        trials[_TRIAL_PROCESS_CLEANUP_FIELD] = \
            compiled["events"][6].address
        trials[_TRIAL_COLLECT_FIELD] = compiled["events"][7].address
        process_handle_count = compiled["process_handle_count"]
        process_width = max(1, process_handle_count)
        runtime_handles = np.zeros(
            (n_trials, process_width), dtype=np.int64)
        runtime_contexts = np.zeros(
            (n_trials, process_width, 2), dtype=np.int64)
        trial_indexes = np.arange(n_trials, dtype=np.int64)
        trials[_PROCESS_DESCRIPTORS_FIELD] = \
            compiled["process_descriptors"].ctypes.data
        trials[_PROCESS_DESCRIPTOR_COUNT_FIELD] = sum(
            not process.spawnable for process in self._processes)
        trials[_PROCESS_HANDLES_FIELD] = (
            runtime_handles.ctypes.data
            + trial_indexes * runtime_handles.strides[0]
        )
        trials[_PROCESS_HANDLE_COUNT_FIELD] = process_handle_count
        trials[_PROCESS_CONTEXTS_FIELD] = (
            runtime_contexts.ctypes.data
            + trial_indexes * runtime_contexts.strides[0]
        )
        trials[_HAS_SPAWNED_FIELD] = int(
            any(process.spawnable for process in self._processes))
        trials[_COLLECT_DESCRIPTORS_FIELD] = \
            compiled["collect_descriptors"].ctypes.data
        trials[_COLLECT_DESCRIPTOR_COUNT_FIELD] = \
            len(compiled["collect_callbacks"])
        trials[_ENTITY_DESCRIPTORS_FIELD] = \
            compiled["entity_descriptors"].ctypes.data
        trials[_ENTITY_DESCRIPTOR_COUNT_FIELD] = \
            compiled["entity_descriptor_count"]
        trials[_RUNTIME_TEXT_HANDLES_FIELD] = \
            compiled["runtime_text_handles"].ctypes.data
        if self._history_captures or self._dataset_captures:
            trials[HISTORY_CAPTURE_TRIAL_FIELD] = np.arange(
                n_trials, dtype=np.uint64)
            trials[HISTORY_CAPTURE_STORE_FIELD] = 0
        # Index each axis directly rather than meshgrid'ing all of them: the
        # design points are a mixed-radix count over the axis sizes, which is
        # exactly meshgrid(indexing="ij") ravel order. Singleton axes cost a
        # broadcast instead of a mesh dimension, so numpy's 32-dimension
        # ceiling no longer applies to the declared parameter count.
        points = np.arange(n_points, dtype=np.int64)
        stride = 1
        for p, axis in zip(reversed(self.params), reversed(axes)):
            size = axis.shape[0]
            if size == 1:
                selected = np.repeat(axis, n_points, axis=0)
            else:
                selected = axis[(points // stride) % size]
            trials[p] = np.repeat(selected, replications, axis=0)
            stride *= size
        for o in self.outputs:
            trials[o] = np.nan
        for field, pred in compiled["preds"].items():
            trials[field] = pred.address
        for field, ev in compiled["user_events"].items():
            trials[field] = ev.address
        for field, index, process_name in compiled["spawn_assignments"]:
            desc = compiled["spawns"][process_name]
            if index is None:
                trials[field] = desc.ctypes.data
            else:
                trials[field][:, index] = desc.ctypes.data

        trials["seed"] = _draw_trial_seeds(seed, n_trials)

        trace_rows: list[np.ndarray] = []
        for tname in self.traces:
            value = param_values[tname]
            shape = self._field_shapes.get(tname)
            if shape is not None and len(shape) != 1:
                raise ValueError(f"trace field '{tname}' has unsupported "
                                 f"shape {shape}")
            slots = 1 if shape is None else shape[0]
            rows = _as_trace_grid(value, trials["seed"], n_trials, slots,
                                  tname)
            trace_rows.extend(row for trial_rows in rows
                              for row in trial_rows)
            field = trials[tname]
            if shape is None:
                for i, trial_rows in enumerate(rows):
                    row = trial_rows[0]
                    field[i, 0] = row.ctypes.data
                    field[i, 1] = row.size
            else:
                for i, trial_rows in enumerate(rows):
                    for slot, row in enumerate(trial_rows):
                        field[i, slot, 0] = row.ctypes.data
                        field[i, slot, 1] = row.size

        swept = tuple(p for p, axis in zip(self.params, axes)
                      if axis.shape[0] > 1)
        return Experiment(self, trials, compiled["trial"].address,
                          keepalive=[*trace_rows, runtime_handles,
                                     runtime_contexts],
                          replications=replications,
                          swept=swept,
                          history_captures=tuple(
                              self._history_captures.values()),
                          dataset_captures=tuple(
                              self._dataset_captures.values()))


class Experiment(Generic[_ExperimentResultT]):
    model: Model[_ExperimentResultT]
    #: One structured record per trial; outputs are filled in by run().
    trials: np.ndarray
    #: Number of failed trials in the last run(), or None before it.
    failures: int | None
    #: Replications per design point (trial order is design-point-major
    #: with replications innermost).
    replications: int
    #: Names of the parameters swept over more than one value.
    swept: tuple[str, ...]
    #: Typed/dynamic access to retained outputs, datasets, and histories.
    results: _ExperimentResultT

    def __init__(self, model: Model[_ExperimentResultT],
                 trials: np.ndarray, trial_addr: int,
                 keepalive: Sequence[np.ndarray] = (),
                 replications: int = 1, swept: Sequence[str] = (),
                 history_captures: Sequence[HistoryCaptureSpec] = (),
                 dataset_captures: Sequence[HistoryCaptureSpec] = ()):
        self.model = model
        self.trials = trials
        self._trial_addr = trial_addr
        # Trace arrays whose data pointers live in the trial records
        self._keepalive = tuple(keepalive)
        self.failures = None
        self.replications = replications
        self.swept = tuple(swept)
        ordered_captures = sorted(history_captures, key=lambda spec: spec.slot)
        ordered_datasets = sorted(dataset_captures, key=lambda spec: spec.slot)
        self._history_capture_specs = tuple(ordered_captures)
        self._dataset_capture_specs = tuple(ordered_datasets)
        self._capture_slot_count = sum(
            spec.slot_count
            for spec in (*self._history_capture_specs,
                         *self._dataset_capture_specs)
        )
        self._history_capture_names = tuple(
            spec.name for spec in self._history_capture_specs)
        self._dataset_capture_names = tuple(
            spec.name for spec in self._dataset_capture_specs)
        self._history_capture_data: dict[str, Any] | None = None
        self._dataset_capture_data: dict[str, Any] | None = None
        # The concrete runtime object is the dynamic fallback. A parameterized
        # Model supplies a narrower static result schema to type checkers.
        self.results = cast(_ExperimentResultT,
                            _build_result_namespace(self))

    def _result_value(self, leaf: _ResultLeaf) -> Any:
        if leaf.family == "outputs":
            return self.trials[leaf.flattened_name]
        if leaf.family == "datasets":
            return self.datasets(leaf.flattened_name)
        if leaf.family == "histories":
            return self.histories(leaf.flattened_name)
        raise ValueError(f"unknown result family: {leaf.family}")

    def run(self) -> int:
        """Run all trials in parallel, in place. Returns the number of
        failed trials (their outputs stay NaN)."""
        trials = self.trials
        if trials.dtype.names is None:
            raise TypeError("experiment must be a structured array")
        if not trials.flags["C_CONTIGUOUS"]:
            raise ValueError("experiment array must be C-contiguous")
        if trials.ndim != 1 or trials.size == 0:
            raise ValueError("experiment must be a non-empty 1-D array")

        fptr = ffi.cast("void(*)(void *)", self._trial_addr)
        buf = ffi.from_buffer(trials, require_writable=True)
        capture_store = ffi.NULL
        if self._capture_slot_count:
            self._history_capture_data = None
            self._dataset_capture_data = None
            capture_store = create_capture_store(
                trials.size, self._capture_slot_count)
            trials[HISTORY_CAPTURE_TRIAL_FIELD] = np.arange(
                trials.size, dtype=np.uint64)
            trials[HISTORY_CAPTURE_STORE_FIELD] = int(
                ffi.cast("intptr_t", capture_store))
        try:
            lib.cimba_run_experiment(buf, trials.size, trials.itemsize, fptr)
            if self._capture_slot_count:
                self._history_capture_data = copy_capture_store(
                    capture_store,
                    num_trials=trials.size,
                    specs=self._history_capture_specs,
                )
                self._dataset_capture_data = copy_capture_store(
                    capture_store,
                    num_trials=trials.size,
                    specs=self._dataset_capture_specs,
                )
        finally:
            if capture_store != ffi.NULL:
                destroy_capture_store(capture_store)
                trials[HISTORY_CAPTURE_STORE_FIELD] = 0

        if not self.model.outputs:
            self.failures = 0
        else:
            failed = np.isnan(self.trials[self.model.outputs[0]])
            if failed.ndim > 1:
                failed = failed.reshape(failed.shape[0], -1).any(axis=1)
            self.failures = int(failed.sum())
        return self.failures

    def summary(self, *outputs: str,
                confidence: float = 0.95) -> np.ndarray:
        """Summarize outputs across replications: a structured array with
        one record per design point, holding the swept parameter values
        and, for each output, its replication mean under its own name and
        the Student-t confidence-interval half-width under
        ``<name>_hw``. With no arguments every output is summarized.

        Failed trials (NaN outputs) are excluded per output; the mean is
        NaN when no trial survived and the half-width is NaN when fewer
        than two did."""
        if self.failures is None:
            raise RuntimeError("run() the experiment before summary()")
        names = list(outputs) if outputs else list(self.model.outputs)
        unknown = set(names) - set(self.model.outputs)
        if unknown:
            raise ValueError(f"unknown outputs: {sorted(unknown)}")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")

        from scipy.stats import t as _student_t

        reps = self.replications
        n_points = self.trials.size // reps
        cols = [(p, self.trials.dtype[p]) for p in self.swept]
        for o in names:
            cols += [(o, self.trials.dtype[o]),
                     (f"{o}_hw", self.trials.dtype[o])]
        table = np.zeros(n_points, dtype=cols)
        for p in self.swept:
            table[p] = self.trials[p][::reps]
        for o in names:
            vals = self.trials[o]
            vals = vals.reshape((n_points, reps) + vals.shape[1:])
            with np.errstate(invalid="ignore", divide="ignore"):
                n = (~np.isnan(vals)).sum(axis=1).astype(np.float64)
                mean = np.nansum(vals, axis=1) / n
                dev = vals - np.expand_dims(mean, 1)
                var = np.nansum(dev * dev, axis=1) / (n - 1.0)
                tcrit = _student_t.ppf((1.0 + confidence) / 2.0, n - 1.0)
                table[o] = mean
                table[f"{o}_hw"] = tcrit * np.sqrt(var / n)
        return table

    def __getitem__(self, field: str) -> np.ndarray:
        return self.trials[field]

    def __len__(self) -> int:
        return self.trials.size

    def histories(
        self,
        name: str,
    ) -> (tuple[np.ndarray, ...]
          | tuple[tuple[np.ndarray, ...], ...]):
        """Captured raw history arrays for every trial.

        Indexed component captures add an inner tuple containing one array
        per collection item in deterministic collection order.
        """
        if name not in self._history_capture_names:
            raise KeyError(f"unknown captured history: {name}")
        if self._history_capture_data is None:
            raise RuntimeError("run() the experiment before reading histories")
        return self._history_capture_data[name]

    def history(
        self,
        name: str,
        *,
        trial: int = 0,
        index: int | None = None,
    ) -> np.ndarray:
        """Captured raw history array for one trial and collection item."""
        rows = self.histories(name)
        if trial < 0 or trial >= len(rows):
            raise IndexError("history trial index out of range")
        spec = next(
            spec for spec in self._history_capture_specs
            if spec.name == name
        )
        if spec.shape is None:
            if index is not None:
                raise TypeError(f"history '{name}' is not indexed")
            return rows[trial]
        if index is None:
            raise TypeError(
                f"history '{name}' requires an index for indexed capture")
        if type(index) is not int:
            raise TypeError("history index must be an int")
        if index < 0 or index >= spec.shape[0]:
            raise IndexError("history collection index out of range")
        return rows[trial][index]

    def datasets(self, name: str) -> tuple[np.ndarray, ...]:
        """Captured raw dataset arrays for every trial."""
        if name not in self._dataset_capture_names:
            raise KeyError(f"unknown captured dataset: {name}")
        if self._dataset_capture_data is None:
            raise RuntimeError("run() the experiment before reading datasets")
        return self._dataset_capture_data[name]

    def dataset(self, name: str, *, trial: int = 0) -> np.ndarray:
        """Captured raw dataset array for one trial."""
        rows = self.datasets(name)
        if trial < 0 or trial >= len(rows):
            raise IndexError("dataset trial index out of range")
        return rows[trial]
