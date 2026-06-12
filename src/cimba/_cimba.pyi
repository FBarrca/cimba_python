"""Type stubs for the compiled ``cimba._cimba`` cffi extension module."""

from typing import Any

from _cffi_backend import FFI

class _Lib:
    """The cdef'd symbols exposed through cffi (see _cffi_build.py)."""

    def cimba_version(self) -> Any: ...  # cdata char *
    def cimba_run_experiment(self, trials: Any, num_trials: int,
                             trial_struct_size: int,
                             trial_func: Any) -> None: ...
    def cmb_random_hwseed(self) -> int: ...
    def cmb_logger_flags_off(self, flags: int) -> None: ...

ffi: FFI
lib: _Lib
