"""Typed helpers for Cimba's top-level cimba module."""

from typing import Any

def native_version() -> str:
    """Return the version string reported by the bundled Cimba C library."""
    ...

def run_native_experiment(
    experiment_buffer: Any,
    trial_struct_size: int,
    trial_func_capsule: object,
) -> None:
    """Run a native Cimba experiment over a writable C-contiguous trial buffer."""
    ...

def set_native_thread_hooks(
    init_capsule: object | None = None,
    user_arg_capsule: object | None = None,
    exit_capsule: object | None = None,
) -> None:
    """Set native Cimba pthread hook capsules."""
    ...
