# This file is included by ../_cimba.pyx.

cdef void *_process_trampoline(cmb_process *me, void *ctx) noexcept with gil:
    cdef Process proc = <Process><object><PyObject *>ctx
    cdef object result
    cdef PyObject *op

    try:
        result = proc._func(proc, proc._context)
        if result is None:
            return NULL
        op = <PyObject *>result
        Py_INCREF(<object>op)
        proc._owns_exit_value = True
        return <void *>op
    except BaseException as exc:
        _record_exception(exc)
        return NULL



def hold(double duration) -> int:
    """Suspend the current process for ``duration`` simulated time units."""
    return cmb_process_hold(duration)


def yield_process() -> int:
    """Yield the current process until another process or timer resumes it."""
    return cmb_process_yield()


def process_exit(object value=None):
    """Exit the current process with an optional Python exit value."""
    cdef Process proc = current_process()
    cdef PyObject *op = NULL
    if value is not None:
        op = <PyObject *>value
        Py_INCREF(<object>op)
        if proc is not None:
            proc._owns_exit_value = True
    cmb_process_exit(<void *>op)


def current_process():
    """Return the Python wrapper for the currently running process, or ``None``."""
    cdef cmb_process *pp = cmb_process_current()
    if pp == NULL:
        return None
    return _process_registry.get(<uintptr_t>pp)


cdef class Process:
    """A Cimba process backed by a Python callable."""

    cdef cmb_process *_ptr
    cdef public bint _closed
    cdef bint _initialized
    cdef bint _keepalive
    cdef bint _owns_exit_value
    cdef object _func
    cdef object _context
    cdef object _name

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True
        self._initialized = False
        self._keepalive = False
        self._owns_exit_value = False
        self._func = None
        self._context = None
        self._name = None

    def __init__(self, str name, object func, object context=None, int priority=0):
        if not callable(func):
            raise TypeError("func must be callable")
        cdef bytes bname = _name_bytes(name)
        self._ptr = cmb_process_create()
        self._func = func
        self._context = context
        self._name = name
        cmb_process_initialize(
            self._ptr,
            bname,
            _process_trampoline,
            <void *><PyObject *>self,
            priority,
        )
        self._initialized = True
        self._closed = False
        _process_registry[<uintptr_t>self._ptr] = self
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
        def __set__(self, int value):
            _raise_if_closed(self)
            cmb_process_priority_set(self._ptr, value)

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
        cmb_process_start(self._ptr)
        return self

    def stop(self) -> int:
        """Stop the process if it is running."""
        _raise_if_closed(self)
        if cmb_process_status(self._ptr) == PROCESS_RUNNING:
            return cmb_process_stop(self._ptr, NULL)
        return STOPPED

    def interrupt(self, int signal=INTERRUPTED, int priority=0) -> None:
        _raise_if_closed(self)
        if signal == SUCCESS:
            raise ValueError("interrupt signal cannot be SUCCESS")
        cmb_process_interrupt(self._ptr, signal, priority)

    def resume(self, int signal=SUCCESS) -> None:
        _raise_if_closed(self)
        cmb_process_resume(self._ptr, signal)

    def wait(self) -> int:
        _raise_if_closed(self)
        return cmb_process_wait_process(self._ptr)

    def timer_add(self, double duration, int signal=TIMEOUT) -> int:
        _raise_if_closed(self)
        return <object>cmb_process_timer_add(self._ptr, duration, signal)

    def timer_set(self, double duration, int signal=TIMEOUT) -> int:
        _raise_if_closed(self)
        return <object>cmb_process_timer_set(self._ptr, duration, signal)

    def timer_cancel(self, int handle) -> bool:
        _raise_if_closed(self)
        return True if cmb_process_timer_cancel(self._ptr, <uint64_t>handle) else False

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
                cmb_process_stop(self._ptr, NULL)
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
            _process_registry.pop(<uintptr_t>self._ptr, None)
            cmb_process_destroy(self._ptr)
            self._ptr = NULL
        if self._keepalive:
            self._keepalive = False
            Py_DECREF(<object><PyObject *>self)
        self._closed = True

