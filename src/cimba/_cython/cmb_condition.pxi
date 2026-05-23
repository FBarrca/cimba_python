# This file is included by ../_cimba.pyx.

cdef bool _condition_demand_trampoline(
    const cmb_condition *cnd,
    const cmb_process *prc,
    const void *ctx,
) noexcept with gil:
    cdef object demand_ctx = <object><PyObject *>ctx
    cdef object predicate = demand_ctx[0]
    cdef object user_ctx = demand_ctx[1]
    cdef Process proc = _process_registry_get().get(<uintptr_t>prc)

    try:
        return True if predicate(proc, user_ctx) else False
    except BaseException as exc:
        _record_exception(exc)
        return False


cdef class Condition:
    """Condition variable with a Python demand predicate."""

    cdef cmb_condition *_ptr
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True

    def __init__(self, str name):
        cdef bytes bname = _name_bytes(name)
        self._ptr = cmb_condition_create()
        cmb_condition_initialize(self._ptr, bname)
        self._closed = False
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    def wait(self, object predicate, object context=None) -> int:
        """Wait until ``predicate(process, context)`` returns true."""
        if self._closed:
            raise RuntimeError("Condition is closed")
        if not callable(predicate):
            raise TypeError("predicate must be callable")
        cdef object demand_ctx = (predicate, context)
        cdef int64_t sig = cmb_condition_wait(
            self._ptr,
            _condition_demand_trampoline,
            <const void *><PyObject *>demand_ctx,
        )
        if sig == _PROCESS_CANCEL_SIGNAL:
            raise _ProcessCancelled()
        return <object>sig

    def signal(self) -> int:
        _raise_if_closed(self)
        return <object>cmb_condition_signal(self._ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            cmb_condition_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True
