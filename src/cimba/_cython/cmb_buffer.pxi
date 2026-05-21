# This file is included by ../_cimba.pyx.

cdef class Buffer:
    """Numeric Cimba buffer with put/get semantics."""

    cdef cmb_buffer *_ptr
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True

    def __init__(self, str name, object capacity=None):
        cdef bytes bname = _name_bytes(name)
        self._ptr = cmb_buffer_create()
        cmb_buffer_initialize(self._ptr, bname, _capacity_to_u64(capacity))
        self._closed = False
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property name:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_buffer_get_name(self._ptr).decode("utf-8")

    property capacity:
        def __get__(self):
            _raise_if_closed(self)
            return <object>self._ptr.capacity

    property level:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_buffer_level(self._ptr)

    property space:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_buffer_space(self._ptr)

    def put(self, int amount=1):
        """Put ``amount`` into the buffer. Returns ``(signal, remaining)``."""
        _raise_if_closed(self)
        cdef uint64_t n = <uint64_t>amount
        cdef int64_t sig = cmb_buffer_put(self._ptr, &n)
        return sig, <object>n

    def get(self, int amount=1):
        """Get ``amount`` from the buffer. Returns ``(signal, obtained)``."""
        _raise_if_closed(self)
        cdef uint64_t n = <uint64_t>amount
        cdef int64_t sig = cmb_buffer_get(self._ptr, &n)
        return sig, <object>n

    def start_recording(self) -> None:
        _raise_if_closed(self)
        cmb_buffer_recording_start(self._ptr)

    def stop_recording(self) -> None:
        _raise_if_closed(self)
        cmb_buffer_recording_stop(self._ptr)

    def history(self):
        _raise_if_closed(self)
        return _timeseries_copy(cmb_buffer_history(self._ptr))

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            cmb_buffer_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True

