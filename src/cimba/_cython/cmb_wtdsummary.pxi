# This file is included by ../_cimba.pyx.

cdef WeightedSummary _wtdsummary_owner(cmb_wtdsummary *ptr):
    cdef WeightedSummary summary = WeightedSummary.__new__(WeightedSummary)
    summary._ptr = ptr
    summary._owner = True
    summary._closed = False
    return summary



cdef class WeightedSummary:
    """Running weighted sample summary."""

    cdef cmb_wtdsummary *_ptr
    cdef bint _owner
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._owner = False
        self._closed = True

    def __init__(self):
        if self._ptr == NULL:
            self._ptr = cmb_wtdsummary_create()
            self._owner = True
            self._closed = False

    def __dealloc__(self):
        if not self._closed:
            self.close()

    def add(self, double value, double weight=1.0) -> int:
        _raise_if_closed(self)
        return <object>cmb_wtdsummary_add(self._ptr, value, weight)

    def reset(self) -> None:
        _raise_if_closed(self)
        cmb_wtdsummary_reset(self._ptr)

    def merge(self, WeightedSummary other):
        _raise_if_closed(self)
        _raise_if_closed(other)
        cdef WeightedSummary merged = WeightedSummary()
        cmb_wtdsummary_merge(merged._ptr, self._ptr, other._ptr)
        return merged

    property count:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_wtdsummary_count(self._ptr)

    property weight_sum:
        def __get__(self):
            _raise_if_closed(self)
            return self._ptr.wsum

    property min:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_min(self._ptr)

    property max:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_max(self._ptr)

    property mean:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_mean(self._ptr)

    property variance:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_variance(self._ptr)

    property stddev:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_stddev(self._ptr)

    property skewness:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_skewness(self._ptr)

    property kurtosis:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_kurtosis(self._ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._owner and self._ptr != NULL:
            cmb_wtdsummary_destroy(self._ptr)
        self._ptr = NULL
        self._closed = True
