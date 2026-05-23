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

    def __cinit__(self):
        self._closed = True
        self._event_initialized = False
        self._random_initialized = False
        self.seed_used = None
        self._owned = []
        self._exception = None

    def __init__(self, object start_time=0.0, object seed=None, bint log_info=False):
        if _active_simulation_get() is not None:
            raise RuntimeError("only one active Simulation is supported per Python thread")

        self._closed = False
        self._event_initialized = False
        self._random_initialized = False
        self._owned = []
        self._exception = None

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
