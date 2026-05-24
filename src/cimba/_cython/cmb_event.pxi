# This file is included by ../_cimba.pyx.

cdef void _stop_at_event(void *subject, void *object) noexcept with gil:
    _cancel_active_processes(None)
    cmb_event_schedule(
        _clear_after_cancellation_event,
        NULL,
        NULL,
        cmb_time(),
        _CLEAR_AFTER_CANCEL_PRIORITY,
    )


@cython.freelist(1024)
cdef class _PythonEventEntry:
    cdef uint64_t handle
    cdef object sim
    cdef object callback
    cdef object subject
    cdef object obj

    def __init__(self, object sim, object callback, object subject, object obj):
        self.handle = 0
        self.sim = sim
        self.callback = callback
        self.subject = subject
        self.obj = obj


cdef void _python_event_trampoline(void *subject, void *object) noexcept with gil:
    cdef PyObject *op = <PyObject *>subject
    cdef _PythonEventEntry entry
    cdef object sim
    cdef object callback
    cdef object event_subject
    cdef object event_obj
    cdef uint64_t handle

    Py_INCREF(<object>op)
    try:
        entry = <_PythonEventEntry><object>op
        sim = entry.sim
        handle = entry.handle
        if sim is not None:
            sim._events.pop(<object>handle, None)
        callback = entry.callback
        event_subject = entry.subject
        event_obj = entry.obj
        callback(event_subject, event_obj)
    except BaseException as exc:
        _record_exception(exc)
    finally:
        Py_DECREF(<object>op)
        Py_DECREF(<object>op)


def time() -> float:
    """Return the current simulation time."""
    return cmb_time()


cdef class Simulation:
    """Own the thread-local Cimba random generator and event queue."""

    cdef public bint _closed
    cdef bint _event_initialized
    cdef bint _random_initialized
    cdef public object seed_used
    cdef public object _owned
    cdef public object _exception
    cdef public object _events

    def __cinit__(self):
        self._closed = True
        self._event_initialized = False
        self._random_initialized = False
        self.seed_used = None
        self._owned = []
        self._exception = None
        self._events = {}

    def __init__(self, object start_time=0.0, object seed=None, bint log_info=False):
        if _active_simulation_get() is not None:
            raise RuntimeError("only one active Simulation is supported per Python thread")

        self._closed = False
        self._event_initialized = False
        self._random_initialized = False
        self._owned = []
        self._exception = None
        self._events = {}

        cdef uint64_t seed_value = cmb_random_hwseed() if seed is None else _seed_to_u64(seed)
        cmb_random_initialize(seed_value)
        self._random_initialized = True
        self.seed_used = <object>seed_value

        cmb_event_queue_initialize(_duration_to_double(start_time, "start_time"))
        self._event_initialized = True
        if log_info:
            cmb_logger_flags_on(LOGGER_INFO)
        else:
            cmb_logger_flags_off(LOGGER_INFO)

        _active_simulation_set(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property closed:
        def __get__(self):
            return True if self._closed else False

    property now:
        def __get__(self):
            return cmb_time()

    property event_count:
        def __get__(self):
            return <object>cmb_event_queue_count()

    property current_event:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_event_current()

    def stop_at(self, object when, object priority=0) -> int:
        """Schedule the event queue to clear at absolute simulation time ``when``."""
        _raise_if_closed(self)
        cdef double stop_time = _duration_to_double(when, "when")
        if stop_time < cmb_time():
            raise ValueError("stop time cannot be before the current simulation time")
        return <object>cmb_event_schedule(
            _stop_at_event,
            NULL,
            NULL,
            stop_time,
            _priority_to_i64(priority),
        )

    def schedule(self, object callback, object when, object subject=None, object obj=None, object priority=0) -> int:
        """Schedule a Python callback at absolute simulation time ``when``."""
        _raise_if_closed(self)
        if not callable(callback):
            raise TypeError("callback must be callable")
        cdef double event_time = _duration_to_double(when, "when")
        if event_time < cmb_time():
            raise ValueError("event time cannot be before the current simulation time")
        cdef int64_t pri = _priority_to_i64(priority)
        cdef _PythonEventEntry entry = _PythonEventEntry(self, callback, subject, obj)
        cdef PyObject *op = <PyObject *>entry
        Py_INCREF(<object>op)
        cdef uint64_t handle = cmb_event_schedule(
            _python_event_trampoline,
            <void *>op,
            NULL,
            event_time,
            pri,
        )
        entry.handle = handle
        self._events[<object>handle] = entry
        return <object>handle

    def schedule_native(
        self,
        object action_capsule,
        object when,
        object subject_capsule=None,
        object object_capsule=None,
        object priority=0,
    ) -> int:
        """Schedule a native Cimba event function from a ``cimba.event_func`` capsule."""
        _raise_if_closed(self)
        cdef cmb_event_func *action = <cmb_event_func *>_capsule_pointer(
            action_capsule,
            _EVENT_FUNC_CAPSULE_NAME,
            "action_capsule",
        )
        cdef void *native_subject = _capsule_payload_pointer(subject_capsule, "subject_capsule")
        cdef void *native_object = _capsule_payload_pointer(object_capsule, "object_capsule")
        cdef double event_time = _duration_to_double(when, "when")
        if event_time < cmb_time():
            raise ValueError("event time cannot be before the current simulation time")
        return <object>cmb_event_schedule(
            action,
            native_subject,
            native_object,
            event_time,
            _priority_to_i64(priority),
        )

    def is_event_scheduled(self, object handle) -> bool:
        _raise_if_closed(self)
        return True if cmb_event_is_scheduled(_handle_to_u64(handle)) else False

    def event_time(self, object handle) -> float:
        _raise_if_closed(self)
        cdef uint64_t h = _handle_to_u64(handle)
        if not cmb_event_is_scheduled(h):
            raise ValueError("event is not scheduled")
        return cmb_event_time(h)

    def event_priority(self, object handle) -> int:
        _raise_if_closed(self)
        cdef uint64_t h = _handle_to_u64(handle)
        if not cmb_event_is_scheduled(h):
            raise ValueError("event is not scheduled")
        return <object>cmb_event_priority(h)

    def cancel_event(self, object handle) -> bool:
        _raise_if_closed(self)
        cdef uint64_t h = _handle_to_u64(handle)
        cdef bint found
        if not cmb_event_is_scheduled(h):
            return False
        found = cmb_event_cancel(h)
        if found:
            self._release_python_event(h)
        return True if found else False

    def reschedule_event(self, object handle, object when) -> bool:
        _raise_if_closed(self)
        cdef uint64_t h = _handle_to_u64(handle)
        cdef double event_time
        if not cmb_event_is_scheduled(h):
            return False
        event_time = _duration_to_double(when, "when")
        if event_time < cmb_time():
            raise ValueError("event time cannot be before the current simulation time")
        return True if cmb_event_reschedule(h, event_time) else False

    def reprioritize_event(self, object handle, object priority) -> bool:
        _raise_if_closed(self)
        cdef uint64_t h = _handle_to_u64(handle)
        if not cmb_event_is_scheduled(h):
            return False
        return True if cmb_event_reprioritize(h, _priority_to_i64(priority)) else False

    def clear(self) -> None:
        """Clear all scheduled events, stopping the simulation run."""
        _raise_if_closed(self)
        if _count_cancellation_events(list(self._owned)) > 0:
            cmb_event_schedule(
                _clear_after_cancellation_event,
                NULL,
                NULL,
                cmb_time(),
                _CLEAR_AFTER_CANCEL_PRIORITY,
            )
        else:
            self._release_python_events()
            cmb_event_queue_clear()

    def execute_next(self) -> bool:
        """Execute one scheduled event."""
        _raise_if_closed(self)
        cdef bool ok = cmb_event_execute_next()
        if self._exception is not None:
            exc = self._exception
            self._exception = None
            _drain_cancellation_events(list(self._owned))
            raise exc
        return True if ok else False

    def execute(self) -> None:
        """Run the event queue until it is empty."""
        _raise_if_closed(self)
        cmb_event_queue_execute()
        if self._exception is not None:
            exc = self._exception
            self._exception = None
            raise exc

    def close(self) -> None:
        """Stop owned processes and release the thread-local Cimba state."""
        cdef object obj
        cdef object owned
        cdef Process proc
        if self._closed:
            return

        owned = list(self._owned)
        if self._event_initialized:
            for obj in owned:
                if isinstance(obj, Process):
                    proc = <Process>obj
                    if proc._ptr != NULL and cmb_process_status(proc._ptr) == PROCESS_RUNNING:
                        proc._request_cancel()
            _drain_cancellation_events(owned)
            self._release_python_events()

        for obj in reversed(owned):
            close = getattr(obj, "close", None)
            if close is not None:
                close()
        self._owned.clear()

        if self._event_initialized:
            cmb_event_queue_terminate()
            self._event_initialized = False
        if self._random_initialized:
            cmb_random_terminate()
            self._random_initialized = False

        if _active_simulation_get() is self:
            _active_simulation_set(None)
        self._closed = True

    cdef void _release_python_event(self, uint64_t handle):
        cdef object entry = self._events.pop(<object>handle, _MISSING)
        if entry is not _MISSING:
            Py_DECREF(entry)

    def _release_python_events(self) -> None:
        cdef object entry
        cdef list entries = list(self._events.values())
        self._events.clear()
        for entry in entries:
            Py_DECREF(entry)
