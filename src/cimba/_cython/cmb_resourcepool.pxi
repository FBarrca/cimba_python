# This file is included by ../_cimba.pyx.

cdef class ResourcePool:
    """Counting semaphore resource pool."""

    cdef cmb_resourcepool *_ptr
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True

    def __init__(self, str name, int capacity):
        cdef bytes bname = _name_bytes(name)
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._ptr = cmb_resourcepool_create()
        cmb_resourcepool_initialize(self._ptr, bname, <uint64_t>capacity)
        self._closed = False
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property name:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_resourcepool_get_name(self._ptr).decode("utf-8")

    property capacity:
        def __get__(self):
            _raise_if_closed(self)
            return <object>self._ptr.capacity

    property in_use:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_resourcepool_in_use(self._ptr)

    property available:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_resourcepool_available(self._ptr)

    def acquire(self, int amount=1) -> int:
        _raise_if_closed(self)
        return cmb_resourcepool_acquire(self._ptr, <uint64_t>amount)

    def preempt(self, int amount=1) -> int:
        _raise_if_closed(self)
        return cmb_resourcepool_preempt(self._ptr, <uint64_t>amount)

    def release(self, int amount=1) -> None:
        _raise_if_closed(self)
        cmb_resourcepool_release(self._ptr, <uint64_t>amount)

    def held_by(self, Process process) -> int:
        _raise_if_closed(self)
        _raise_if_closed(process)
        return <object>cmb_resourcepool_held_by_process(self._ptr, process._ptr)

    def start_recording(self) -> None:
        _raise_if_closed(self)
        cmb_resourcepool_start_recording(self._ptr)

    def stop_recording(self) -> None:
        _raise_if_closed(self)
        cmb_resourcepool_stop_recording(self._ptr)

    def history(self):
        _raise_if_closed(self)
        return _timeseries_copy(cmb_resourcepool_get_history(self._ptr))

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            cmb_resourcepool_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True

