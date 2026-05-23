# This file is included by ../_cimba.pyx.

cdef DataSummary _datasummary_owner(cmb_datasummary *ptr):
    cdef DataSummary summary = DataSummary.__new__(DataSummary)
    summary._ptr = ptr
    summary._owner = True
    summary._closed = False
    return summary



cdef class DataSummary:
    """Running unweighted sample summary."""

    cdef cmb_datasummary *_ptr
    cdef bint _owner
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._owner = False
        self._closed = True

    def __init__(self):
        if self._ptr == NULL:
            self._ptr = cmb_datasummary_create()
            self._owner = True
            self._closed = False

    def __dealloc__(self):
        if not self._closed:
            self.close()

    def add(self, double value) -> int:
        _raise_if_closed(self)
        return <object>cmb_datasummary_add(self._ptr, value)

    def reset(self) -> None:
        _raise_if_closed(self)
        cmb_datasummary_reset(self._ptr)

    def merge(self, DataSummary other):
        _raise_if_closed(self)
        _raise_if_closed(other)
        cdef DataSummary merged = DataSummary()
        cmb_datasummary_merge(merged._ptr, self._ptr, other._ptr)
        return merged

    property count:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_datasummary_count(self._ptr)

    property min:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_min(self._ptr)

    property max:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_max(self._ptr)

    property mean:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_mean(self._ptr)

    property variance:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_variance(self._ptr)

    property stddev:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_stddev(self._ptr)

    property skewness:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_skewness(self._ptr)

    property kurtosis:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_kurtosis(self._ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._owner and self._ptr != NULL:
            cmb_datasummary_destroy(self._ptr)
        self._ptr = NULL
        self._closed = True
