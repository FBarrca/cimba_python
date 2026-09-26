"""Native trial runtime: the fixed lifecycle ABI and callback compilation.

Every trial record starts with the same header (``_LIFECYCLE_ABI_FIELDS``):
the standard timing/seed fields, then the addresses of the fixed lifecycle
callbacks and of per-model descriptor tables. The lifecycle functions below
are compiled once per process and read everything model-specific from those
tables -- which entities to create (``_ENTITY_DESCRIPTOR_*`` rows), which
processes to start (``_PROCESS_DESCRIPTOR_*`` rows), which collectors to
call -- so no source is generated per record layout.

``_compile_cfuncs`` turns Python functions into native callbacks, compiling
batches in a fork pool when that is safe and serially otherwise; only the
fixed lifecycle callbacks are cached across models.
"""

import hashlib
import threading
from collections.abc import Callable, Iterable, Sequence
from types import FunctionType
from typing import Any, cast
from uuid import uuid4

import numpy as np
from numba import carray, cfunc, from_dtype, njit, types
from numba.core.cpu import CPUContext
from numba.core.ccallback import CFunc
from numba.core.dispatcher import Dispatcher

from . import _bindings as _b
from ._cimba import lib
from ._declarations import _STANDARD_FIELDS
from ._intrinsics import addressof, call_void_callback, ptr_caster

_LIFECYCLE_FIELDS = (
    "_cimba_recording_event _cimba_trial_initialize _cimba_trial_entities "
    "_cimba_trial_processes _cimba_trial_teardown _cimba_trial_stop "
    "_cimba_trial_process_cleanup _cimba_trial_collect "
    "_cimba_process_descriptors _cimba_process_descriptor_count "
    "_cimba_process_handles _cimba_process_handle_count "
    "_cimba_process_contexts _cimba_has_spawned _cimba_collect_descriptors "
    "_cimba_collect_descriptor_count _cimba_entity_descriptors "
    "_cimba_entity_descriptor_count "
    "_cimba_trial_completed"
).split()
(
    _RECORDING_EVENT_FIELD,
    _TRIAL_INITIALIZE_FIELD,
    _TRIAL_ENTITIES_FIELD,
    _TRIAL_PROCESSES_FIELD,
    _TRIAL_TEARDOWN_FIELD,
    _TRIAL_STOP_FIELD,
    _TRIAL_PROCESS_CLEANUP_FIELD,
    _TRIAL_COLLECT_FIELD,
    _PROCESS_DESCRIPTORS_FIELD,
    _PROCESS_DESCRIPTOR_COUNT_FIELD,
    _PROCESS_HANDLES_FIELD,
    _PROCESS_HANDLE_COUNT_FIELD,
    _PROCESS_CONTEXTS_FIELD,
    _HAS_SPAWNED_FIELD,
    _COLLECT_DESCRIPTORS_FIELD,
    _COLLECT_DESCRIPTOR_COUNT_FIELD,
    _ENTITY_DESCRIPTORS_FIELD,
    _ENTITY_DESCRIPTOR_COUNT_FIELD,
    _TRIAL_COMPLETED_FIELD,
) = _LIFECYCLE_FIELDS

_LIFECYCLE_ABI_FIELDS = [
    *_STANDARD_FIELDS,
    *((name, "<i8") for name in _LIFECYCLE_FIELDS),
]
_LIFECYCLE_ABI_DTYPE = np.dtype(_LIFECYCLE_ABI_FIELDS)
_LIFECYCLE_ABI_RECORD = from_dtype(_LIFECYCLE_ABI_DTYPE)
_LIFECYCLE_ABI_PTR = types.CPointer(_LIFECYCLE_ABI_RECORD)
_INT64_FROM_ADDRESS = ptr_caster(types.int64)
_FLOAT64_FROM_ADDRESS = ptr_caster(types.float64)


_PROCESS_DESCRIPTOR_WIDTH = 8
(
    _PD_CALLBACK,
    _PD_NAME,
    _PD_ALLOC_SIZE,
    _PD_PRIORITY,
    _PD_COPIES,
    _PD_INDEXED,
    _PD_HANDLE_START,
    _PD_DEST_OFFSET,
) = range(_PROCESS_DESCRIPTOR_WIDTH)

_ENTITY_DESCRIPTOR_WIDTH = 5
(_ED_KIND, _ED_NAME, _ED_CAPACITY_MODE, _ED_CAPACITY, _ED_DEST_OFFSET) = range(
    _ENTITY_DESCRIPTOR_WIDTH
)
(
    _ENTITY_BUFFER,
    _ENTITY_RESOURCE,
    _ENTITY_RESOURCEPOOL,
    _ENTITY_OBJECTQUEUE,
    _ENTITY_DATASET,
    _ENTITY_CONDITION,
    _ENTITY_PRIORITYQUEUE,
) = range(1, 8)
_CAPACITY_CONSTANT, _CAPACITY_FIELD = range(2)


@njit(inline="always")
def _runtime_entity_table(vtrl):
    env = carray(vtrl, 1)[0]
    count = env[_ENTITY_DESCRIPTOR_COUNT_FIELD]
    descriptors = carray(
        _INT64_FROM_ADDRESS(env[_ENTITY_DESCRIPTORS_FIELD]),
        max(1, count) * _ENTITY_DESCRIPTOR_WIDTH,
    )
    return env, addressof(vtrl), count, descriptors


@njit(inline="always")
def _runtime_entity_handle(self_addr, descriptors, base):
    return carray(
        _INT64_FROM_ADDRESS(self_addr + descriptors[base + _ED_DEST_OFFSET]), 1
    )[0]


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


def _native_cfunc(signature: Any) -> Callable[[Callable[..., Any]], CFunc]:
    """Build a callback without an unused Python-callable wrapper."""
    return cfunc(signature, no_cpython_wrapper=True)


def _compile_parallel_cfunc(signature: Any,
                            function: Callable[..., Any]) -> CFunc:
    """Compile a C boundary that turns Numba exceptions into trial failure.

    Numba's default cfunc wrapper prints and discards an exception. Catch it
    while still in native code, then enter Cimba's native trial recovery.
    """
    # Forked compilers inherit Numba's symbol counter. Give each compilation
    # a unique symbol so separately loaded libraries cannot bind to an older
    # model's identically named callback with the same record layout.
    symbol = f"_cimba_callback_{uuid4().hex}"
    owned = FunctionType(function.__code__, function.__globals__, symbol,
                         function.__defaults__, function.__closure__)
    owned.__qualname__ = symbol
    inner = njit(signature)(owned)
    assert isinstance(inner, Dispatcher)
    arguments = ", ".join(f"a{i}" for i in range(len(signature.args)))
    fallback = "None" if signature.return_type == types.void else "0"
    # Numba infers `none` for a body that always raises, even with an explicit
    # return signature. Such a body has no successful value to forward.
    if inner.nopython_signatures[0].return_type == types.void:
        call = f"inner({arguments})\n        return {fallback}"
    else:
        call = f"return inner({arguments})"
    namespace = {"inner": inner, "abandon": _b.trial_abandon}
    exec(
        f"def {symbol}_guarded({arguments}):\n"
        f"    try:\n        {call}\n"
        f"    except Exception:\n        pass\n"
        f"    abandon()\n    return {fallback}\n", namespace)
    return _native_cfunc(signature)(namespace[f"{symbol}_guarded"])


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
        # The scheduled stop event normally handled this already, but user
        # code can clear the event queue and end a trial early.  Cimba requires
        # a process to be stopped before it is terminated.
        if _b.process_status(handles[index]) == 1:
            _b.process_stop(handles[index], 0)
        _b.process_terminate(handles[index])
        _b.process_destroy(handles[index])
    if env[_HAS_SPAWNED_FIELD] != 0:
        _b.spawned_stop_all()
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
    env, self_addr, count, descriptors = _runtime_entity_table(vtrl)
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
    env, self_addr, count, descriptors = _runtime_entity_table(subject)
    for index in range(count):
        base = index * _ENTITY_DESCRIPTOR_WIDTH
        kind = descriptors[base + _ED_KIND]
        handle = _runtime_entity_handle(self_addr, descriptors, base)
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
    if env["duration_s"] == np.inf:
        # Quiescence mode has no automatic recording window or stop event.
        return
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
    env, self_addr, count, descriptors = _runtime_entity_table(vtrl)
    for index in range(count):
        base = index * _ENTITY_DESCRIPTOR_WIDTH
        kind = descriptors[base + _ED_KIND]
        handle = _runtime_entity_handle(self_addr, descriptors, base)
        # RC2 requires initialize/terminate and create/destroy pairs.
        if kind == _ENTITY_BUFFER:
            _b.buffer_terminate(handle)
            _b.buffer_destroy(handle)
        elif kind == _ENTITY_RESOURCE:
            _b.resource_terminate(handle)
            _b.resource_destroy(handle)
        elif kind == _ENTITY_RESOURCEPOOL:
            _b.resourcepool_terminate(handle)
            _b.resourcepool_destroy(handle)
        elif kind == _ENTITY_OBJECTQUEUE:
            _b.objectqueue_terminate(handle)
            _b.objectqueue_destroy(handle)
        elif kind == _ENTITY_DATASET:
            _b.dataset_terminate(handle)
            _b.dataset_destroy(handle)
        elif kind == _ENTITY_CONDITION:
            _b.condition_terminate(handle)
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
    env[_TRIAL_COMPLETED_FIELD] = 1


_LIFECYCLE_JOBS = (
    (types.void(_LIFECYCLE_ABI_PTR, types.intp), _runtime_recording_event),
    *(
        (types.void(_LIFECYCLE_ABI_PTR), fn)
        for fn in (
            _runtime_trial_initialize,
            _runtime_trial_entities,
            _runtime_trial_processes,
            _runtime_trial_teardown,
            _runtime_trial,
            _runtime_trial_stop,
            _runtime_trial_process_cleanup,
            _runtime_trial_collect,
        )
    ),
)


class _LoadedCFunc:
    """A callback from a fork-compiled library loaded into the parent."""

    __slots__ = ("address", "library", "native_name")

    def __init__(self, library, native_name: str):
        self.library = library
        self.native_name = native_name
        self.address = library.get_pointer_to_function(native_name)


def _cpu_context() -> CPUContext:
    """Numba's CPU target context (a lazily created cached property that
    type checkers see as the descriptor itself)."""
    from numba.core.registry import cpu_target

    return cast(CPUContext, cpu_target.target_context)


def _load_compiled_library(state):
    """Load one worker's linked object-code library into the parent."""
    from numba.core.runtime import nrt

    context = _cpu_context()
    nrt.rtsys.initialize(context)
    return context.codegen().unserialize_library(state)


_PARALLEL_CFUNC_JOBS: tuple[tuple[Any, Callable[..., Any]], ...] = ()


def _compile_cfunc_job(index: int) -> tuple[int, Any, str]:
    """Compile a callback for this worker's parent process only.

    Captured text handles point to buffers kept alive in the parent and
    inherited by fork. The returned object code must not be persisted or
    loaded into another interpreter.
    """
    signature, function = _PARALLEL_CFUNC_JOBS[index]
    callback = _compile_parallel_cfunc(signature, function)
    native_name = callback.native_name
    assert native_name is not None
    return (
        index,
        callback._library.serialize_using_object_code(),
        native_name,
    )


def _compile_uncached_cfuncs(
    jobs: Sequence[tuple[Any, Callable[..., Any]]],
) -> list[Any]:
    """Compile callbacks in a fork pool, with a portable serial fallback."""
    if len(jobs) < 2:
        return [_compile_parallel_cfunc(signature, function)
                for signature, function in jobs]
    try:
        import multiprocessing

        # Forking from a non-main Python thread is unsafe, and the inherited
        # job table below is intentionally single-owner. Keep that uncommon
        # path functional with serial compilation.
        if (threading.current_thread() is not threading.main_thread()
                or threading.active_count() != 1):
            raise ValueError

        context = multiprocessing.get_context("fork")
    except (ImportError, ValueError):
        return [_compile_parallel_cfunc(signature, function)
                for signature, function in jobs]
    # Populate Numba's registries and native runtime before forking.  These
    # tables are copy-on-write state, so every compiler worker can inherit the
    # setup instead of repeating it on its first callback.

    from numba.core.runtime import nrt
    cpu_context = _cpu_context()
    cpu_context.refresh()
    nrt.rtsys.initialize(cpu_context)
    cpu_context.codegen()

    global _PARALLEL_CFUNC_JOBS
    _PARALLEL_CFUNC_JOBS = tuple(jobs)
    try:
        with context.Pool(processes=min(6, len(jobs))) as pool:
            serialized = pool.map(_compile_cfunc_job, range(len(jobs)))
    finally:
        _PARALLEL_CFUNC_JOBS = ()
    callbacks: list[Any] = [None] * len(jobs)
    for local_index, state, native_name in serialized:
        library = _load_compiled_library(state)
        callbacks[local_index] = _LoadedCFunc(library, native_name)
    return callbacks


# Only these fixed library callbacks are shared across models. User code has
# no fingerprint or persistent cache: its owner is the concrete Model instance.
_LIFECYCLE_CALLBACKS: dict[Callable[..., Any], Any] = {}


def _compile_cfuncs(jobs: Sequence[tuple[Any, Callable[..., Any]]]) -> list[Any]:
    cached = dict(_LIFECYCLE_CALLBACKS)
    missing = [(signature, function) for signature, function in jobs
               if function not in cached]
    compiled = iter(_compile_uncached_cfuncs(missing))
    fixed = {function for _signature, function in _LIFECYCLE_JOBS}
    callbacks = []
    for _signature, function in jobs:
        callback = cached.get(function)
        if callback is None:
            callback = next(compiled)
            if function in fixed:
                _LIFECYCLE_CALLBACKS[function] = callback
        callbacks.append(callback)
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
