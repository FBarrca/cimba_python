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
        cdef Py_ssize_t count = <Py_ssize_t>self._ptr.count
        cdef Py_ssize_t i
        cdef list result = [None] * count
        for i in range(count):
            result[i] = self._ptr.xa[i]
        return result

    def summary(self):
        _raise_if_closed(self)
        cdef cmb_datasummary *summary = cmb_datasummary_create()
        cmb_dataset_summarize(self._ptr, summary)
        return _datasummary_owner(summary)

    def reset(self) -> None:
        _raise_if_closed(self)
        cmb_dataset_reset(self._ptr)

    def copy(self):
        _raise_if_closed(self)
        cdef Dataset copied = Dataset()
        cmb_dataset_copy(copied._ptr, self._ptr)
        return copied

    def merge(self, Dataset other):
        _raise_if_closed(self)
        _raise_if_closed(other)
        cdef Dataset merged = Dataset()
        cdef uint64_t i
        cmb_dataset_copy(merged._ptr, self._ptr)
        for i in range(other._ptr.count):
            cmb_dataset_add(merged._ptr, other._ptr.xa[i])
        return merged

    def sort(self) -> None:
        _raise_if_closed(self)
        cmb_dataset_sort(self._ptr)

    def acf(self, object lags):
        _raise_if_closed(self)
        cdef uint16_t n = _lags_to_u16(lags)
        cdef double *values
        cdef list result
        cdef uint16_t i
        if self._ptr.count < 2:
            raise ValueError("at least two samples are required")
        if n == 0:
            return [1.0]
        if n >= self._ptr.count:
            raise ValueError("lags must be smaller than sample count")
        values = <double *>PyMem_Malloc((<size_t>n + 1) * sizeof(double))
        if values == NULL:
            raise MemoryError()
        try:
            cmb_dataset_ACF(self._ptr, <unsigned int>n, values)
            result = [0.0] * (n + 1)
            with cython.boundscheck(False):
                for i in range(n + 1):
                    result[i] = values[i]
            return result
        finally:
            PyMem_Free(values)

    def pacf(self, object lags):
        _raise_if_closed(self)
        cdef uint16_t n = _lags_to_u16(lags)
        cdef double *values
        cdef list result
        cdef uint16_t i
        if self._ptr.count < 3:
            raise ValueError("at least three samples are required")
        if n == 0:
            return [1.0]
        if n >= self._ptr.count - 1:
            raise ValueError("lags must be smaller than sample count minus one")
        values = <double *>PyMem_Malloc((<size_t>n + 1) * sizeof(double))
        if values == NULL:
            raise MemoryError()
        try:
            cmb_dataset_PACF(self._ptr, <unsigned int>n, values, NULL)
            result = [0.0] * (n + 1)
            with cython.boundscheck(False):
                for i in range(n + 1):
                    result[i] = values[i]
            return result
        finally:
            PyMem_Free(values)

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
