"""Cython exports and CFFI access to the same native runtime."""

from . import _cimba_native as _native
from cffi import FFI

for _name in dir(_native):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_native, _name)

ffi = FFI()
ffi.cdef("""
    const char *cimba_version(void);
    uint64_t cimba_run(void *your_experiment_array,
                              uint64_t num_trials,
                              size_t trial_struct_size,
                              void (*your_trial_func)(void *));
    uint32_t cimba_threads_use(uint32_t n_threads);
    uint64_t cmb_random_hwseed(void);
    uint64_t cpy_process_sizeof(void);
    void *cpy_history_capture_store_create(uint64_t num_trials,
                                           uint64_t num_slots);
    void cpy_history_capture_store_destroy(void *store);
    uint64_t cpy_history_capture_store_count(const void *store,
                                             uint64_t trial,
                                             uint64_t slot);
    const double *cpy_history_capture_store_data(const void *store,
                                                 uint64_t trial,
                                                 uint64_t slot);
""")
lib = ffi.dlopen(_native.__file__)


__all__ = [name for name in dir(_native) if not name.startswith("__")]
__all__.extend(["ffi", "lib"])
