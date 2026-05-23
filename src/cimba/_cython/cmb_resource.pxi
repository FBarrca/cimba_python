# This file is included by ../_cimba.pyx.

cdef class Resource:
    """Binary semaphore resource."""

    cdef cmb_resource *_ptr
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True

    def __init__(self, str name):
        cdef bytes bname = _name_bytes(name)
        self._ptr = cmb_resource_create()
        cmb_resource_initialize(self._ptr, bname)
        self._closed = False
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property name:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_resource_name(self._ptr).decode("utf-8")

    property in_use:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_resource_in_use(self._ptr)

    property available:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_resource_available(self._ptr)

    def acquire(self) -> int:
        if self._closed:
            raise RuntimeError("Resource is closed")
        cdef int64_t sig = cmb_resource_acquire(self._ptr)
        if sig == _PROCESS_CANCEL_SIGNAL:
            raise _ProcessCancelled()
        return <object>sig

    def preempt(self) -> int:
        if self._closed:
            raise RuntimeError("Resource is closed")
        cdef int64_t sig = cmb_resource_preempt(self._ptr)
        if sig == _PROCESS_CANCEL_SIGNAL:
            raise _ProcessCancelled()
        return <object>sig

    def release(self) -> None:
        _raise_if_closed(self)
        cmb_resource_release(self._ptr)

    def held_by(self, Process process) -> int:
        _raise_if_closed(self)
        _raise_if_closed(process)
        return <object>cmb_resource_held_by_process(self._ptr, process._ptr)

    def start_recording(self) -> None:
        _raise_if_closed(self)
        cmb_resource_start_recording(self._ptr)

    def stop_recording(self) -> None:
        _raise_if_closed(self)
        cmb_resource_stop_recording(self._ptr)

    def history(self):
        _raise_if_closed(self)
        return _timeseries_copy(cmb_resource_history(self._ptr))

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            cmb_resource_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True
