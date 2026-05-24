# This file is included by ../_cimba.pyx.

cdef void *_process_result_pointer(Process proc, object result):
    cdef PyObject *op
    if result is None:
        return NULL
    op = <PyObject *>result
    Py_INCREF(<object>op)
    proc._owns_exit_value = True
    return <void *>op


cdef void *_process_trampoline(cmb_process *me, void *ctx) noexcept with gil:
    cdef Process proc = <Process><object><PyObject *>ctx
    cdef object result

    try:
        result = proc._func(proc, proc._context)
        return _process_result_pointer(proc, result)
    except _ProcessExit as exc:
        return _process_result_pointer(proc, exc.value)
    except _ProcessCancelled:
        return NULL
    except BaseException as exc:
        _record_exception(exc)
        return NULL



def hold(double duration) -> int:
    """Suspend the current process for ``duration`` simulated time units."""
    cdef int64_t sig
    if not isfinite(duration):
        raise ValueError("duration must be finite")
    if duration < 0.0:
        raise ValueError("duration must be non-negative")
    sig = cmb_process_hold(duration)
    if sig == _PROCESS_CANCEL_SIGNAL:
        raise _ProcessCancelled()
    return <object>sig


def yield_process() -> int:
    """Yield the current process until another process or timer resumes it."""
    cdef int64_t sig = cmb_process_yield()
    if sig == _PROCESS_CANCEL_SIGNAL:
        raise _ProcessCancelled()
    return <object>sig


def wait_event(object handle) -> int:
    """Yield the current process until a scheduled event fires or is canceled."""
    cdef object sim = _active_simulation_get()
    cdef uint64_t h
    cdef int64_t sig
    if sim is None or sim._closed:
        raise RuntimeError("wait_event requires an active Simulation")
    if cmb_process_current() == NULL:
        raise RuntimeError("wait_event must be called from a running process")
    h = _handle_to_u64(handle)
    if not cmb_event_is_scheduled(h):
        raise ValueError("event is not scheduled")
    sig = cmb_process_wait_event(h)
    if sig == _PROCESS_CANCEL_SIGNAL:
        raise _ProcessCancelled()
    return <object>sig


def process_exit(object value=None):
    """Exit the current process with an optional Python exit value."""
    raise _ProcessExit(value)


def current_process():
    """Return the Python wrapper for the currently running process, or ``None``."""
    cdef cmb_process *pp = cmb_process_current()
    if pp == NULL:
        return None
    return _process_registry_get().get(<uintptr_t>pp)


cdef class Process:
    """A Cimba process backed by a Python callable."""

    cdef cmb_process *_ptr
    cdef public bint _closed
    cdef bint _initialized
    cdef bint _keepalive
    cdef bint _owns_exit_value
    cdef bint _cancelling
    cdef object _func
    cdef object _context
    cdef object _name

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True
        self._initialized = False
        self._keepalive = False
        self._owns_exit_value = False
        self._cancelling = False
        self._func = None
        self._context = None
        self._name = None

    def __init__(self, str name, object func, object context=None, object priority=0):
        if not callable(func):
            raise TypeError("func must be callable")
        cdef bytes bname = _name_bytes(name)
        cdef int64_t pri = _priority_to_i64(priority)
        self._ptr = cmb_process_create()
        self._func = func
        self._context = context
        self._name = name
        cmb_process_initialize(
            self._ptr,
            bname,
            _process_trampoline,
            <void *><PyObject *>self,
            pri,
        )
        self._initialized = True
        self._closed = False
        _process_registry_get()[<uintptr_t>self._ptr] = self
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property name:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_process_name(self._ptr).decode("utf-8")

    property priority:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_process_priority(self._ptr)
        def __set__(self, object value):
            _raise_if_closed(self)
            cmb_process_priority_set(self._ptr, _priority_to_i64(value))

    property status:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_process_status(self._ptr)

    def start(self):
        """Schedule the process to start at the current simulation time."""
        _raise_if_closed(self)
        if not self._keepalive:
            Py_INCREF(<object><PyObject *>self)
            self._keepalive = True
        self._cancelling = False
        cmb_process_start(self._ptr)
        return self

    cdef int64_t _request_cancel(self) except *:
        cdef cmb_process *current = cmb_process_current()
        if current == self._ptr:
            self._cancelling = True
            raise _ProcessCancelled()
        if cmb_process_status(self._ptr) == PROCESS_RUNNING:
            self._cancelling = True
            cmb_process_resume(self._ptr, _PROCESS_CANCEL_SIGNAL)
            return SUCCESS
        return STOPPED

    def stop(self) -> int:
        """Cooperatively stop the process if it is running."""
        _raise_if_closed(self)
        return <object>self._request_cancel()

    def interrupt(self, object signal=INTERRUPTED, object priority=0) -> None:
        _raise_if_closed(self)
        cdef int64_t sig = _signal_to_i64(signal)
        if sig == SUCCESS:
            raise ValueError("interrupt signal cannot be SUCCESS")
        cmb_process_interrupt(self._ptr, sig, _priority_to_i64(priority))

    def resume(self, object signal=SUCCESS) -> None:
        _raise_if_closed(self)
        cmb_process_resume(self._ptr, _signal_to_i64(signal))

    def wait(self) -> int:
        _raise_if_closed(self)
        cdef int64_t sig = cmb_process_wait_process(self._ptr)
        if sig == _PROCESS_CANCEL_SIGNAL:
            raise _ProcessCancelled()
        return <object>sig

    def timer_add(self, object duration, object signal=TIMEOUT) -> int:
        _raise_if_closed(self)
        return <object>cmb_process_timer_add(
            self._ptr,
            _duration_to_double(duration, "duration"),
            _signal_to_i64(signal),
        )

    def timer_set(self, object duration, object signal=TIMEOUT) -> int:
        _raise_if_closed(self)
        return <object>cmb_process_timer_set(
            self._ptr,
            _duration_to_double(duration, "duration"),
            _signal_to_i64(signal),
        )

    def timer_cancel(self, object handle) -> bool:
        _raise_if_closed(self)
        return True if cmb_process_timer_cancel(self._ptr, _handle_to_u64(handle)) else False

    def timers_clear(self) -> None:
        _raise_if_closed(self)
        cmb_process_timers_clear(self._ptr)

    def exit_value(self):
        _raise_if_closed(self)
        cdef void *ptr = cmb_process_exit_value(self._ptr)
        if ptr == NULL:
            return None
        cdef object obj = <object><PyObject *>ptr
        return obj

    def close(self) -> None:
        cdef void *ptr = NULL
        if self._closed:
            return
        if self._ptr != NULL:
            if cmb_process_status(self._ptr) == PROCESS_RUNNING:
                self._request_cancel()
                if cmb_process_status(self._ptr) == PROCESS_RUNNING:
                    raise RuntimeError("cannot close a running process before cooperative cancellation has executed")
            cmb_event_pattern_cancel(
                CMB_ANY_ACTION,
                <const void *>self._ptr,
                <const void *>CMB_ANY_OBJECT,
            )
            if self._owns_exit_value:
                ptr = cmb_process_exit_value(self._ptr)
                if ptr != NULL:
                    Py_DECREF(<object><PyObject *>ptr)
                self._owns_exit_value = False
            if self._initialized:
                cmb_process_terminate(self._ptr)
                self._initialized = False
            _process_registry_get().pop(<uintptr_t>self._ptr, None)
            cmb_process_destroy(self._ptr)
            self._ptr = NULL
        if self._keepalive:
            self._keepalive = False
            Py_DECREF(<object><PyObject *>self)
        self._closed = True


cdef uint64_t _count_cancellation_events_for_process(Process proc):
    if proc._ptr == NULL:
        return 0
    return cmb_event_pattern_count(
        CMB_ANY_ACTION,
        <const void *>proc._ptr,
        <const void *>_PROCESS_CANCEL_SIGNAL,
    )


cdef uint64_t _count_cancellation_events(object processes):
    cdef uint64_t count = 0
    cdef object obj
    for obj in processes:
        if isinstance(obj, Process):
            count += _count_cancellation_events_for_process(<Process>obj)
    return count


cdef void _drain_cancellation_events(object processes):
    cdef uint64_t pending = _count_cancellation_events(processes)
    cdef uint64_t guard = pending + cmb_event_queue_count() + 1
    while pending > 0 and guard > 0:
        if not cmb_event_execute_next():
            break
        pending = _count_cancellation_events(processes)
        guard -= 1
