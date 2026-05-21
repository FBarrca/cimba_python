# This file is included by ../_cimba.pyx.

cdef class Dataset:
    """Resizable unweighted data set of doubles."""

    cdef cmb_dataset *_ptr
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True

    def __init__(self):
        self._ptr = cmb_dataset_create()
        self._closed = False

    def __dealloc__(self):
        if not self._closed:
            self.close()

    def add(self, double value) -> int:
        _raise_if_closed(self)
        return <object>cmb_dataset_add(self._ptr, value)

    def values(self):
        _raise_if_closed(self)
        return [self._ptr.xa[i] for i in range(self._ptr.count)]

    def summary(self):
        _raise_if_closed(self)
        cdef cmb_datasummary *summary = cmb_datasummary_create()
        cmb_dataset_summarize(self._ptr, summary)
        return _datasummary_owner(summary)

    property count:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_dataset_count(self._ptr)

    property min:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_dataset_min(self._ptr)

    property max:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_dataset_max(self._ptr)

    property median:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_dataset_median(self._ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            cmb_dataset_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True

