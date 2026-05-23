# This file is included by ../_cimba.pyx.

cdef class ObjectQueue:
    """FIFO queue for Python objects."""

    cdef cmb_objectqueue *_ptr
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True

    def __init__(self, str name, object capacity=None):
        cdef bytes bname = _name_bytes(name)
        self._ptr = cmb_objectqueue_create()
        cmb_objectqueue_initialize(self._ptr, bname, _capacity_to_u64(capacity))
        self._closed = False
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property name:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_objectqueue_name(self._ptr).decode("utf-8")

    property capacity:
        def __get__(self):
            _raise_if_closed(self)
            return <object>self._ptr.capacity

    property length:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_objectqueue_length(self._ptr)

    property space:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_objectqueue_space(self._ptr)

    def put(self, object obj) -> int:
        if self._closed:
            raise RuntimeError("ObjectQueue is closed")
        cdef PyObject *op = <PyObject *>obj
        Py_INCREF(<object>op)
        cdef int64_t sig = cmb_objectqueue_put(self._ptr, <void *>op)
        if sig != SUCCESS:
            Py_DECREF(<object>op)
            if sig == _PROCESS_CANCEL_SIGNAL:
                raise _ProcessCancelled()
        return <object>sig

    def get(self):
        """Get one object. Returns ``(signal, object)``."""
        if self._closed:
            raise RuntimeError("ObjectQueue is closed")
        cdef void *ptr = NULL
        cdef int64_t sig = cmb_objectqueue_get(self._ptr, &ptr)
        if sig != SUCCESS:
            if sig == _PROCESS_CANCEL_SIGNAL:
                raise _ProcessCancelled()
            return sig, None
        return sig, _object_from_owned_pointer(ptr)

    def position(self, object obj) -> int:
        _raise_if_closed(self)
        return <object>cmb_objectqueue_position(self._ptr, <void *><PyObject *>obj)

    def start_recording(self) -> None:
        _raise_if_closed(self)
        cmb_objectqueue_recording_start(self._ptr)

    def stop_recording(self) -> None:
        _raise_if_closed(self)
        cmb_objectqueue_recording_stop(self._ptr)

    def history(self):
        _raise_if_closed(self)
        return _timeseries_copy(cmb_objectqueue_history(self._ptr))

    cdef void _decref_queued_objects(self):
        cdef void *ptr = NULL
        while self._ptr != NULL and cmb_objectqueue_length(self._ptr) > 0:
            ptr = NULL
            if cmb_objectqueue_get(self._ptr, &ptr) == SUCCESS and ptr != NULL:
                Py_DECREF(<object><PyObject *>ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            self._decref_queued_objects()
            cmb_objectqueue_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True
