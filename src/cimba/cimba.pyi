"""Typed helpers for Cimba's top-level cimba module."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

def native_version() -> str:
    """Return the version string reported by the bundled Cimba C library."""
    ...

def gil_enabled() -> bool:
    """Return whether this interpreter runs with the GIL enabled.

    ``run_experiment(..., backend="thread")`` parallelizes only when this
    returns ``False`` (a free-threaded build); otherwise thread replications run
    serially.
    """
    ...

def run_experiment(
    trial_fn: Callable[[int, int | None], Any],
    n: int | None = None,
    *,
    seed: int | None = None,
    seeds: Sequence[int | None] | None = None,
    backend: Literal["process", "thread"] = "process",
    processes: int | None = None,
) -> list[Any]:
    """Run independent replications of ``trial_fn``.

    ``trial_fn(index, seed)`` is called once per replication and its return value
    is collected into the result list at ``index``. The default ``"process"``
    backend uses forked worker processes; ``"thread"`` uses Cimba's native
    pthread worker pool.
    """
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
