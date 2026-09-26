"""Numba operations on typed record handles, independent of input sources."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from typing import Any, cast as type_cast

from llvmlite import ir
from numba import carray, from_dtype, types
from numba.extending import intrinsic, overload, overload_method

from cimba.engine.symbols import register
from cimba.layout import MODEL_RECORD_CLASSES, STORE_MODEL_TARGETS, RecordLayout
from cimba.modeling import (
    Process, end_trial, hold, log, now, release, schedule, suspend, this_process,
)
from cimba.schema import ClassSchema
from .numba_compat import cgutils, infer_getattr, make_overload_method_template
from .views import (
    EventHandleType, PredicateHandleType, PROCESS_HANDLE,
    ProcessHandleType, SCHEDULED_HANDLE, ScheduledHandleType,
    register_views,
)

for _name in ("cmb_process_hold", "cmb_time", "cpy_buffer_put",
              "cpy_buffer_get", "cpy_buffer_mean_level", "cpy_buffer_level",
              "cpy_buffer_history", "cpy_timeseries_max", "cimba_trial_abandon",
              "cmb_resourcepool_acquire", "cmb_resourcepool_preempt",
              "cmb_resourcepool_held", "cmb_resourcepool_release",
              "cpy_resourcepool_available",
              "cpy_resourcepool_mean_in_use", "cmb_dataset_add",
              "cpy_dataset_mean", "cpy_dataset_count", "cpy_dataset_max",
              "cpy_store_put_int", "cpy_store_get_int",
              "cpy_store_put_float", "cpy_store_get_float",
              "cpy_store_put_model", "cpy_store_get_model",
              "cpy_priority_put_int", "cpy_priority_get_int",
              "cpy_priority_put_float", "cpy_priority_get_float",
              "cpy_priority_put_model", "cpy_priority_get_model",
              "cpy_priority_enqueue_model", "cpy_priority_cancel",
              "cpy_priority_position", "cpy_priority_length",
              "cpy_store_length",
              "cpy_condition_wait", "cpy_condition_signal",
              "cpy_schedule_event", "cpy_schedule_cancel",
              "cpy_schedule_pending", "cpy_model_release",
              "cpy_process_current", "cpy_process_yield",
              "cpy_process_status", "cpy_process_timer_set",
              "cmb_process_resume", "cmb_process_interrupt",
              "cmb_process_priority_set",
              "cmb_process_timers_clear",
              "cpy_end_trial", "cpy_logger_user_msg", "cpy_logger_user_f64"):
    register(_name)

_native_hold = types.ExternalFunction("cmb_process_hold", types.int64(types.float64))
_native_time = types.ExternalFunction("cmb_time", types.float64())
_native_container_put = types.ExternalFunction(
    "cpy_buffer_put", types.int64(types.intp, types.uint64))
_native_container_get = types.ExternalFunction(
    "cpy_buffer_get", types.int64(types.intp, types.uint64))
_native_container_mean = types.ExternalFunction(
    "cpy_buffer_mean_level", types.float64(types.intp))
_native_container_level = types.ExternalFunction(
    "cpy_buffer_level", types.uint64(types.intp))
_native_container_history = types.ExternalFunction(
    "cpy_buffer_history", types.intp(types.intp))
_native_timeseries_max = types.ExternalFunction(
    "cpy_timeseries_max", types.float64(types.intp))
_native_abandon = types.ExternalFunction("cimba_trial_abandon", types.void())
_native_resource_acquire = types.ExternalFunction(
    "cmb_resourcepool_acquire", types.int64(types.intp, types.uint64))
_native_resource_preempt = types.ExternalFunction(
    "cmb_resourcepool_preempt", types.int64(types.intp, types.uint64))
_native_resource_held = types.ExternalFunction(
    "cmb_resourcepool_held", types.uint64(types.intp, types.intp))
_native_resource_available = types.ExternalFunction(
    "cpy_resourcepool_available", types.uint64(types.intp))
_native_resource_release = types.ExternalFunction(
    "cmb_resourcepool_release", types.void(types.intp, types.uint64))
_native_resource_mean = types.ExternalFunction(
    "cpy_resourcepool_mean_in_use", types.float64(types.intp))
_native_dataset_add = types.ExternalFunction(
    "cmb_dataset_add", types.uint64(types.intp, types.float64))
_native_dataset_mean = types.ExternalFunction(
    "cpy_dataset_mean", types.float64(types.intp))
_native_dataset_count = types.ExternalFunction(
    "cpy_dataset_count", types.uint64(types.intp))
_native_dataset_max = types.ExternalFunction(
    "cpy_dataset_max", types.float64(types.intp))
_store_put_int = types.ExternalFunction(
    "cpy_store_put_int", types.int64(types.intp, types.int64))
_store_get_int = types.ExternalFunction(
    "cpy_store_get_int", types.int64(types.intp))
_store_put_float = types.ExternalFunction(
    "cpy_store_put_float", types.int64(types.intp, types.float64))
_store_get_float = types.ExternalFunction(
    "cpy_store_get_float", types.float64(types.intp))
_store_put_model = types.ExternalFunction(
    "cpy_store_put_model", types.int64(types.intp, types.intp))
_store_get_model = types.ExternalFunction(
    "cpy_store_get_model", types.intp(types.intp))
_priority_put_int = types.ExternalFunction(
    "cpy_priority_put_int", types.int64(types.intp, types.int64, types.int64))
_priority_get_int = types.ExternalFunction(
    "cpy_priority_get_int", types.int64(types.intp))
_priority_put_float = types.ExternalFunction(
    "cpy_priority_put_float", types.int64(types.intp, types.float64, types.int64))
_priority_get_float = types.ExternalFunction(
    "cpy_priority_get_float", types.float64(types.intp))
_priority_put_model = types.ExternalFunction(
    "cpy_priority_put_model", types.int64(types.intp, types.intp, types.int64))
_priority_get_model = types.ExternalFunction(
    "cpy_priority_get_model", types.intp(types.intp))
_priority_enqueue_model = types.ExternalFunction(
    "cpy_priority_enqueue_model", types.uint64(types.intp, types.intp,
                                                types.int64))
_priority_cancel = types.ExternalFunction(
    "cpy_priority_cancel", types.uint64(types.intp, types.uint64))
_priority_position = types.ExternalFunction(
    "cpy_priority_position", types.uint64(types.intp, types.uint64))
_priority_length = types.ExternalFunction(
    "cpy_priority_length", types.uint64(types.intp))
_store_length = types.ExternalFunction(
    "cpy_store_length", types.uint64(types.intp))
_condition_wait = types.ExternalFunction(
    "cpy_condition_wait", types.int64(types.intp, types.intp, types.intp))
_condition_signal = types.ExternalFunction(
    "cpy_condition_signal", types.uint64(types.intp))
_schedule_event = types.ExternalFunction(
    "cpy_schedule_event", types.uint64(
        types.intp, types.intp, types.float64, types.int64))
_event_cancel = types.ExternalFunction(
    "cpy_schedule_cancel", types.uint64(types.uint64))
_event_pending = types.ExternalFunction(
    "cpy_schedule_pending", types.uint64(types.uint64))
_model_release = types.ExternalFunction(
    "cpy_model_release", types.void(types.intp))
_process_current = types.ExternalFunction(
    "cpy_process_current", types.intp())
_process_yield = types.ExternalFunction(
    "cpy_process_yield", types.int64())
_process_status = types.ExternalFunction(
    "cpy_process_status", types.int64(types.intp))
_process_timer_set = types.ExternalFunction(
    "cpy_process_timer_set", types.uint64(types.intp, types.float64,
                                         types.int64))
_process_resume = types.ExternalFunction(
    "cmb_process_resume", types.void(types.intp, types.int64))
_process_interrupt = types.ExternalFunction(
    "cmb_process_interrupt", types.void(types.intp, types.int64,
                                        types.int64))
_process_priority_set = types.ExternalFunction(
    "cmb_process_priority_set", types.void(types.intp, types.int64))
_process_timers_clear = types.ExternalFunction(
    "cmb_process_timers_clear", types.void(types.intp))
_end_trial = types.ExternalFunction("cpy_end_trial", types.void())
_log_message = types.ExternalFunction(
    "cpy_logger_user_msg", types.void(types.uint32, types.intp))
_log_value = types.ExternalFunction(
    "cpy_logger_user_f64", types.void(types.uint32, types.intp, types.float64))
_log_buffers = {}


@overload(log, prefer_literal=True)
def model_log(flags, message, value=None):
    if not isinstance(message, types.StringLiteral):
        return None
    buffer = _log_buffers.setdefault(
        message.literal_value,
        ctypes.create_string_buffer(message.literal_value.encode("utf-8")),
    )
    address = ctypes.addressof(buffer)
    if value is None or isinstance(value, (types.NoneType, types.Omitted)):
        def implementation(flags, message, value=None):
            _log_message(flags, address)
    else:
        def implementation(flags, message, value=None):
            _log_value(flags, address, float(value))  # pyright: ignore[reportArgumentType]
    return implementation


@intrinsic
def scheduled_from_id(typing_context, handle):
    if not isinstance(handle, types.Integer):
        return None

    def codegen(context, builder, signature, arguments):
        proxy = cgutils.create_struct_proxy(SCHEDULED_HANDLE)(context, builder)
        proxy.handle = context.cast(builder, arguments[0], handle, types.uint64)
        return proxy._getvalue()

    return SCHEDULED_HANDLE(handle), codegen


@intrinsic
def process_from_pointer(typing_context, pointer):
    if not isinstance(pointer, types.Integer):
        return None

    def codegen(context, builder, signature, arguments):
        proxy = cgutils.create_struct_proxy(PROCESS_HANDLE)(context, builder)
        proxy.pointer = context.cast(builder, arguments[0], pointer, types.intp)
        return proxy._getvalue()

    return PROCESS_HANDLE(pointer), codegen


@intrinsic
def record_address(typing_context, value):
    if not isinstance(value, types.Record):
        raise TypeError("expected a model or handle record")

    def codegen(context, builder, signature, arguments):
        return builder.ptrtoint(arguments[0], context.get_value_type(types.intp))

    return types.intp(value), codegen


@intrinsic
def native_input_next(typing_context, address):
    if not isinstance(address, types.Integer):
        raise TypeError("input address must be an integer")
    register("cpy_input_next")

    def codegen(context, builder, signature, arguments):
        fnty = ir.FunctionType(ir.DoubleType(), [context.get_value_type(types.intp)])
        fn = cgutils.get_or_insert_function(builder.module, fnty,
                                            "cpy_input_next")
        return builder.call(fn, [arguments[0]])

    return types.float64(address), codegen


@intrinsic
def native_series_at(typing_context, address, time):
    if not isinstance(address, types.Integer) or not isinstance(time, types.Float):
        raise TypeError("series requires an address and float time")
    register("cpy_series_at")

    def codegen(context, builder, signature, arguments):
        fnty = ir.FunctionType(ir.DoubleType(),
                               [context.get_value_type(types.intp), ir.DoubleType()])
        fn = cgutils.get_or_insert_function(builder.module, fnty,
                                            "cpy_series_at")
        return builder.call(fn, arguments)

    return types.float64(address, time), codegen


@intrinsic
def native_series_now(typing_context, address):
    if not isinstance(address, types.Integer):
        raise TypeError("series address must be an integer")
    register("cpy_series_now")

    def codegen(context, builder, signature, arguments):
        fnty = ir.FunctionType(ir.DoubleType(), [context.get_value_type(types.intp)])
        fn = cgutils.get_or_insert_function(builder.module, fnty,
                                            "cpy_series_now")
        return builder.call(fn, arguments)

    return types.float64(address), codegen


# Numba intrinsics are invoked with compiled arguments, not their Python
# typing-function signatures exposed by static analysis.
_input_next_draw = type_cast(Any, native_input_next)
_series_at_draw = type_cast(Any, native_series_at)
_series_now_draw = type_cast(Any, native_series_now)
_process_handle = type_cast(Any, process_from_pointer)
_scheduled_handle = type_cast(Any, scheduled_from_id)


def _record_method(name, predicate):
    """Register a method only on its handle record kind.

    Numba's ordinary ``overload_method(types.Record, ...)`` claims the method
    name on *every* record, including unrelated model fields with that name.
    Restricting attribute resolution here keeps handle methods transparent to
    ordinary record attributes.
    """
    def decorate(function):
        base = make_overload_method_template(
            types.Record, name, function, inline="never")

        class HandleMethod(base):
            def _resolve(self, record, attr):  # pyright: ignore[reportIncompatibleMethodOverride]
                if not predicate(record):
                    return None
                return super()._resolve(record, attr)

        infer_getattr(HandleMethod)
        overload(function)(function)
        return function
    return decorate


def _input_slot(record) -> bool:
    return isinstance(record, types.Record) and all(
        name in record.fields for name in ("kind", "stream_state", "cursor", "step"))


@_record_method("next", _input_slot)
def input_next(record):
    if _input_slot(record):
        if "__cimba_input_int" in record.fields:
            def next_int(record):
                value = _input_next_draw(record_address(record))
                if record.status != 0:
                    _native_abandon()
                return int(value)
            return next_int
        if "__cimba_input_bool" in record.fields:
            def next_bool(record):
                value = _input_next_draw(record_address(record))
                if record.status != 0:
                    _native_abandon()
                return bool(value)
            return next_bool
        def next_float(record):
            value = _input_next_draw(record_address(record))
            if record.status != 0:
                _native_abandon()
            return value
        return next_float


@_record_method("at", _input_slot)
def series_at(record, time):
    if _input_slot(record):
        if "__cimba_input_int" in record.fields:
            def at_int(record, time):
                value = _series_at_draw(record_address(record), float(time))
                if record.status != 0:
                    _native_abandon()
                return int(value)
            return at_int
        if "__cimba_input_bool" in record.fields:
            def at_bool(record, time):
                value = _series_at_draw(record_address(record), float(time))
                if record.status != 0:
                    _native_abandon()
                return bool(value)
            return at_bool
        def at_float(record, time):
            value = _series_at_draw(record_address(record), float(time))
            if record.status != 0:
                _native_abandon()
            return value
        return at_float


@_record_method("now", _input_slot)
def series_now(record):
    if _input_slot(record):
        if "__cimba_input_int" in record.fields:
            def now_int(record):
                value = _series_now_draw(record_address(record))
                if record.status != 0:
                    _native_abandon()
                return int(value)
            return now_int
        if "__cimba_input_bool" in record.fields:
            def now_bool(record):
                value = _series_now_draw(record_address(record))
                if record.status != 0:
                    _native_abandon()
                return bool(value)
            return now_bool
        def now_float(record):
            value = _series_now_draw(record_address(record))
            if record.status != 0:
                _native_abandon()
            return value
        return now_float


@_record_method("remaining", _input_slot)
def input_remaining(record):
    if _input_slot(record):
        def implementation(record):
            if record.kind == 1:
                return -1
            if record.cursor >= record.length:
                return 0
            return int(record.length - record.cursor)
        return implementation


@overload(hold)
def model_hold(duration):
    def implementation(duration):
        return _native_hold(float(duration))
    return implementation


@overload(Process)
def process_from_address(address):
    if isinstance(address, types.Integer):
        def implementation(address):
            return process_from_pointer(address)
        return implementation


@overload(now)
def model_now():
    def implementation():
        return _native_time()
    return implementation


@overload(this_process)
def model_this_process():
    def implementation():
        pointer = _process_current()
        if pointer == 0:
            _native_abandon()
        return _process_handle(pointer)
    return implementation


@overload(suspend)
def model_suspend():
    def implementation():
        return _process_yield()
    return implementation


@overload(end_trial)
def model_end_trial():
    def implementation():
        _end_trial()
    return implementation


@overload_method(ProcessHandleType, "status")
def process_status(handle):
    def implementation(handle):
        return _process_status(handle.pointer)
    return implementation


@overload_method(ProcessHandleType, "timer_set")
def process_timer_set(handle, delay, signal):
    def implementation(handle, delay, signal):
        return _process_timer_set(handle.pointer, float(delay), int(signal))
    return implementation


@overload_method(ProcessHandleType, "resume")
def process_resume(handle, signal=0):
    def implementation(handle, signal=0):
        _process_resume(handle.pointer, int(signal))
    return implementation


@overload_method(ProcessHandleType, "interrupt")
def process_interrupt(handle, signal=-1, priority=0):
    def implementation(handle, signal=-1, priority=0):
        _process_interrupt(handle.pointer, int(signal), int(priority))
    return implementation


@overload_method(ProcessHandleType, "priority_set")
def process_priority_set(handle, priority):
    def implementation(handle, priority):
        _process_priority_set(handle.pointer, int(priority))
    return implementation


@overload_method(ProcessHandleType, "timers_clear")
def process_timers_clear(handle):
    def implementation(handle):
        _process_timers_clear(handle.pointer)
    return implementation


@overload(schedule)
def model_schedule(event, delay, priority=0):
    if isinstance(event, EventHandleType):
        def implementation(event, delay, priority=0):
            return _scheduled_handle(_schedule_event(
                event.callback, event.context, float(delay), int(priority)))
        return implementation


@overload(release)
def model_release(model):
    if isinstance(model, types.Record):
        def implementation(model):
            _model_release(record_address(model))
        return implementation


@overload_method(ScheduledHandleType, "cancel")
def scheduled_cancel(handle):
    def implementation(handle):
        return bool(_event_cancel(handle.handle))
    return implementation


@overload_method(ScheduledHandleType, "pending")
def scheduled_pending(handle):
    def implementation(handle):
        return bool(_event_pending(handle.handle))
    return implementation


def _container_handle(record) -> bool:
    return isinstance(record, types.Record) and "container_handle" in record.fields


_QUEUE_FIELDS = ("container_handle", "store_int_handle",
                 "store_float_handle", "priority_int_handle",
                 "priority_float_handle", "store_model_handle",
                 "priority_model_handle")


def _queue_handle(record) -> bool:
    return isinstance(record, types.Record) and any(
        name in record.fields for name in _QUEUE_FIELDS)


def _model_store_target(record):
    for marker, model_class in STORE_MODEL_TARGETS.items():
        if marker in record.fields:
            register_views(ClassSchema.of(model_class))
            return model_class
    return None


def _record_model_class(record):
    if not isinstance(record, types.Record):
        return None
    for marker, model_class in MODEL_RECORD_CLASSES.items():
        if marker in record.fields:
            return model_class
    return None


@lru_cache(maxsize=512)
def _model_pointer_cast(record_type):
    pointer_type = types.CPointer(record_type)

    @intrinsic
    def cast(typing_context, address):
        if not isinstance(address, types.Integer):
            return None

        def codegen(context, builder, signature, arguments):
            return builder.inttoptr(
                arguments[0], context.get_value_type(pointer_type))

        return pointer_type(address), codegen

    return cast


@_record_method("put", _queue_handle)
def queue_put(record, value, priority=0):
    if _container_handle(record):
        def implementation(record, value, priority=0):
            _native_container_put(record.container_handle, int(value))
        return implementation
    if "store_int_handle" in record.fields:
        def implementation(record, value, priority=0):
            _store_put_int(record.store_int_handle, int(value))
        return implementation
    if "store_float_handle" in record.fields:
        def implementation(record, value, priority=0):
            _store_put_float(record.store_float_handle, float(value))
        return implementation
    if "priority_int_handle" in record.fields:
        def implementation(record, value, priority=0):
            _priority_put_int(record.priority_int_handle, int(value), int(priority))
        return implementation
    if "priority_float_handle" in record.fields:
        def implementation(record, value, priority=0):
            _priority_put_float(record.priority_float_handle, float(value), int(priority))
        return implementation
    if "store_model_handle" in record.fields:
        actual = _record_model_class(value)
        target = _model_store_target(record)
        if actual is None or target is None or not issubclass(actual, target):
            raise TypeError("Store value must be a model of its declared type")
        def implementation(record, value, priority=0):
            _store_put_model(record.store_model_handle, record_address(value))
        return implementation
    if "priority_model_handle" in record.fields:
        actual = _record_model_class(value)
        target = _model_store_target(record)
        if actual is None or target is None or not issubclass(actual, target):
            raise TypeError("PriorityStore value must be a model of its declared type")
        def implementation(record, value, priority=0):
            _priority_put_model(record.priority_model_handle,
                                record_address(value), int(priority))
        return implementation


def _priority_handle(record):
    return isinstance(record, types.Record) and any(
        name in record.fields for name in (
            "priority_int_handle", "priority_float_handle",
            "priority_model_handle"))


@_record_method("enqueue", _priority_handle)
def priority_enqueue(record, value, priority=0):
    if "priority_model_handle" in record.fields:
        actual = _record_model_class(value)
        target = _model_store_target(record)
        if actual is None or target is None or not issubclass(actual, target):
            raise TypeError("PriorityStore value must match its model type")
        def implementation(record, value, priority=0):
            return _priority_enqueue_model(record.priority_model_handle,
                                           record_address(value), int(priority))
        return implementation


@_record_method("cancel", _priority_handle)
def priority_cancel(record, handle):
    if "priority_model_handle" in record.fields:
        def implementation(record, handle):
            return bool(_priority_cancel(record.priority_model_handle,
                                         int(handle)))
        return implementation


@_record_method("position", _priority_handle)
def priority_position(record, handle):
    if "priority_model_handle" in record.fields:
        def implementation(record, handle):
            return _priority_position(record.priority_model_handle, int(handle))
        return implementation


@_record_method("length", _queue_handle)
def queue_length(record):
    if "priority_model_handle" in record.fields:
        def implementation(record):
            return _priority_length(record.priority_model_handle)
        return implementation
    if "store_model_handle" in record.fields:
        def implementation(record):
            return _store_length(record.store_model_handle)
        return implementation


@_record_method("get", _queue_handle)
def queue_get(record, amount=0):
    if _container_handle(record):
        def get_container(record, amount=0):
            _native_container_get(record.container_handle, int(amount))
        return get_container
    if "store_int_handle" in record.fields:
        def get_store_int(record, amount=0):
            return _store_get_int(record.store_int_handle)
        return get_store_int
    if "store_float_handle" in record.fields:
        def get_store_float(record, amount=0):
            return _store_get_float(record.store_float_handle)
        return get_store_float
    if "priority_int_handle" in record.fields:
        def get_priority_int(record, amount=0):
            return _priority_get_int(record.priority_int_handle)
        return get_priority_int
    if "priority_float_handle" in record.fields:
        def get_priority_float(record, amount=0):
            return _priority_get_float(record.priority_float_handle)
        return get_priority_float
    if "store_model_handle" in record.fields:
        target = _model_store_target(record)
        item_type = from_dtype(RecordLayout.of(ClassSchema.of(target)).dtype)
        cast = type_cast(Any, _model_pointer_cast(item_type))
        def get_store_model(record, amount=0):
            return carray(cast(_store_get_model(record.store_model_handle)), 1)[0]
        return get_store_model
    if "priority_model_handle" in record.fields:
        target = _model_store_target(record)
        item_type = from_dtype(RecordLayout.of(ClassSchema.of(target)).dtype)
        cast = type_cast(Any, _model_pointer_cast(item_type))
        def get_priority_model(record, amount=0):
            return carray(cast(_priority_get_model(record.priority_model_handle)), 1)[0]
        return get_priority_model


@_record_method("mean_level", _container_handle)
def container_mean_level(record):
    if _container_handle(record):
        def implementation(record):
            return _native_container_mean(record.container_handle)
        return implementation


@_record_method("max_level", _container_handle)
def container_max_level(record):
    if _container_handle(record):
        def implementation(record):
            return _native_timeseries_max(
                _native_container_history(record.container_handle))
        return implementation


@_record_method("level", _container_handle)
def container_level(record):
    if _container_handle(record):
        def implementation(record):
            return _native_container_level(record.container_handle)
        return implementation


def _resource_handle(record) -> bool:
    return isinstance(record, types.Record) and "resource_handle" in record.fields


@_record_method("acquire", _resource_handle)
def resource_acquire(record, amount=1):
    if _resource_handle(record):
        def implementation(record, amount=1):
            return _native_resource_acquire(record.resource_handle, int(amount))
        return implementation


@_record_method("preempt", _resource_handle)
def resource_preempt(record, amount=1):
    if _resource_handle(record):
        def implementation(record, amount=1):
            return _native_resource_preempt(record.resource_handle, int(amount))
        return implementation


@_record_method("held", _resource_handle)
def resource_held(record, process):
    if _resource_handle(record) and isinstance(process, ProcessHandleType):
        def implementation(record, process):
            return _native_resource_held(record.resource_handle, process.pointer)
        return implementation


@_record_method("release", _resource_handle)
def resource_release(record, amount=1):
    if _resource_handle(record):
        def implementation(record, amount=1):
            _native_resource_release(record.resource_handle, int(amount))
        return implementation


@_record_method("mean_in_use", _resource_handle)
def resource_mean(record):
    if _resource_handle(record):
        def implementation(record):
            return _native_resource_mean(record.resource_handle)
        return implementation


@_record_method("available", _resource_handle)
def resource_available(record):
    if _resource_handle(record):
        def implementation(record):
            return _native_resource_available(record.resource_handle)
        return implementation


def _dataset_handle(record) -> bool:
    return isinstance(record, types.Record) and "dataset_handle" in record.fields


@_record_method("record", _dataset_handle)
def dataset_record(record, value):
    if _dataset_handle(record):
        def implementation(record, value):
            _native_dataset_add(record.dataset_handle, float(value))
        return implementation


@_record_method("sample_mean", _dataset_handle)
def dataset_mean(record):
    if _dataset_handle(record):
        def implementation(record):
            return _native_dataset_mean(record.dataset_handle)
        return implementation


@_record_method("sample_count", _dataset_handle)
def dataset_count(record):
    if _dataset_handle(record):
        def implementation(record):
            return _native_dataset_count(record.dataset_handle)
        return implementation


@_record_method("sample_max", _dataset_handle)
def dataset_max(record):
    if _dataset_handle(record):
        def implementation(record):
            return _native_dataset_max(record.dataset_handle)
        return implementation


def _condition_handle(record) -> bool:
    return isinstance(record, types.Record) and "condition_handle" in record.fields


@_record_method("wait_until", _condition_handle)
def condition_wait(record, predicate):
    if isinstance(predicate, PredicateHandleType):
        def implementation(record, predicate):
            _condition_wait(record.condition_handle, predicate.callback,
                            predicate.context)
        return implementation


@_record_method("signal", _condition_handle)
def condition_signal(record):
    def implementation(record):
        _condition_signal(record.condition_handle)
    return implementation
