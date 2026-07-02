"""Facade over Cimba's CFFI and Cython native extension modules."""

from . import _cimba_native as _native
from cffi import FFI

for _name in dir(_native):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_native, _name)

ffi = FFI()
ffi.cdef("""
    const char *cimba_version(void);
    void cimba_run_experiment(void *your_experiment_array,
                              uint64_t num_trials,
                              size_t trial_struct_size,
                              void (*your_trial_func)(void *));
    uint64_t cmb_random_hwseed(void);
    uint64_t cpy_process_sizeof(void);
    void cpy_logger_flags_on(uint32_t flags);
    void cpy_logger_flags_off(uint32_t flags);
    void cmb_logger_flags_off(uint32_t flags);
""")
lib = ffi.dlopen(_native.__file__)


def logger_flags_on(flags: int) -> None:
    """Turn on native logger flags in both native extension runtimes."""
    _native.logger_flags_on(flags)
    lib.cpy_logger_flags_on(flags)


def logger_flags_off(flags: int) -> None:
    """Turn off native logger flags in both native extension runtimes."""
    _native.logger_flags_off(flags)
    lib.cpy_logger_flags_off(flags)


__all__ = [name for name in dir(_native) if not name.startswith("__")]
__all__.extend(["ffi", "lib", "logger_flags_on", "logger_flags_off"])
