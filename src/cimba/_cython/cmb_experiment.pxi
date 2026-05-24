# This file is included by ../_cimba.pyx.
#
# Parallel experiment orchestration for Python-defined simulations.
#
# The default process backend forks Python worker processes and runs each trial
# in a separate interpreter. The thread backend below preserves Cimba's native
# pthread execution model for in-process/native-object use cases.
#
# Cimba's C runtime is fully thread-local (event queue, coroutine scheduler,
# clock, RNG, mempools), so each worker pthread spawned by cimba_run_experiment
# runs a completely independent simulation. The only thing serializing
# Python-defined trials is the interpreter lock, so we bridge Python callables
# into the native pthread pool through a `noexcept nogil` trampoline that
# re-enters Python with `with gil:`. On a free-threaded interpreter
# (python3.13t / python3.14t) those re-entries run concurrently, giving real
# cross-core parallelism; on a standard GIL build the same code is correct but
# runs the trials serially.

import sys
import warnings
import multiprocessing
from array import array

# Module-level experiment context. Written once (under the GIL) by
# run_experiment before releasing the GIL into cimba_run_experiment, read by the
# worker threads while the run is in flight, then cleared. _experiment_lock
# serializes overlapping run_experiment calls from different Python threads
# (cimba_run_experiment itself also guards its globals with a mutex).
cdef object _experiment_lock = threading.Lock()
cdef object _exp_trial_fn = None
cdef object _exp_seeds = None
cdef object _exp_results = None
cdef object _exp_exceptions = None

# Process-backend context. Written immediately before forking the worker pool so
# forked children inherit the user's trial function without pickling it.
cdef object _process_experiment_lock = threading.Lock()
cdef object _process_trial_fn = None
cdef object _process_seeds = None


cdef void _run_python_trial(uint64_t idx) noexcept:
    # Runs with the GIL held (one worker thread's tstate). Any exception is
    # captured per-trial rather than allowed to escape the noexcept trampoline;
    # run_experiment re-raises the first one after the run completes.
    cdef object trial_fn = _exp_trial_fn
    cdef object seeds = _exp_seeds
    cdef object seed_value
    try:
        seed_value = None if seeds is None else seeds[idx]
        _exp_results[idx] = trial_fn(<object>idx, seed_value)
    except BaseException as exc:
        _exp_exceptions[idx] = exc


cdef void _py_trial_trampoline(void *trial) noexcept nogil:
    cdef uint64_t idx = (<uint64_t *>trial)[0]
    with gil:
        _run_python_trial(idx)


def _process_trial_worker(object idx):
    cdef object seeds = _process_seeds
    cdef object seed_value = None if seeds is None else seeds[idx]
    return _process_trial_fn(idx, seed_value)


def gil_enabled() -> bool:
    """Return whether this interpreter is running with the GIL enabled.

    ``run_experiment(..., backend="thread")`` only achieves real parallelism
    when this returns ``False`` (a free-threaded build, e.g. ``python3.13t``).
    On a standard interpreter it returns ``True`` and the thread backend runs
    trials serially.
    """
    cdef object fn = getattr(sys, "_is_gil_enabled", None)
    if fn is None:
        return True
    return True if fn() else False


cdef list _resolve_seeds(object n, object seed, object seeds):
    cdef object base
    cdef uint64_t num
    cdef uint64_t i
    if seeds is not None:
        if seed is not None:
            raise ValueError("pass either seed or seeds, not both")
        seed_list = [None if s is None else _seed_to_u64(s) for s in seeds]
        if n is not None and _u64_value(n, "n", 0) != <uint64_t>len(seed_list):
            raise ValueError("n must equal len(seeds) when both are given")
        return seed_list
    if n is None:
        raise ValueError("n is required unless seeds is given")
    num = _u64_value(n, "n", 0)
    base = cmb_random_hwseed() if seed is None else _seed_to_u64(seed)
    return [<object>cmb_random_fmix64(<uint64_t>base, i) for i in range(num)]


cdef object _validate_processes(object processes):
    if processes is None:
        return None
    return <object>_u64_value(processes, "processes", 1)


cdef list _run_process_experiment(object trial_fn, list seed_list, object processes):
    global _process_trial_fn, _process_seeds
    cdef uint64_t num_trials = <uint64_t>len(seed_list)
    cdef object ctx
    cdef object worker_indexes
    cdef object worker_processes
    if num_trials == 0:
        return []
    worker_indexes = range(num_trials)
    worker_processes = _validate_processes(processes)

    with _process_experiment_lock:
        _process_trial_fn = trial_fn
        _process_seeds = seed_list
        try:
            try:
                ctx = multiprocessing.get_context("fork")
            except ValueError as exc:
                raise RuntimeError(
                    "backend='process' requires multiprocessing start method 'fork'; "
                    "use backend='thread' on platforms without fork"
                ) from exc

            with ctx.Pool(processes=worker_processes) as pool:
                return pool.map(_process_trial_worker, worker_indexes)
        finally:
            _process_trial_fn = None
            _process_seeds = None


cdef list _run_thread_experiment(object trial_fn, list seed_list):
    global _exp_trial_fn, _exp_seeds, _exp_results, _exp_exceptions
    cdef uint64_t num_trials = <uint64_t>len(seed_list)
    if num_trials == 0:
        return []

    if num_trials > 1 and gil_enabled():
        # Python's default warning filter shows this once per call site, so
        # ordinary loops stay quiet after the first run.
        warnings.warn(
            "run_experiment backend='thread' runs replications serially under "
            "a GIL-enabled interpreter; use backend='process' or a free-threaded "
            "build (e.g. python3.13t) for parallelism.",
            RuntimeWarning,
            stacklevel=2,
        )

    cdef object buf = array("Q", range(num_trials))
    cdef object byte_mv = memoryview(buf).cast("B")
    cdef unsigned char[::1] view = byte_mv
    cdef list results
    cdef list exceptions
    cdef object exc

    with _experiment_lock:
        _exp_trial_fn = trial_fn
        _exp_seeds = seed_list
        _exp_results = [None] * num_trials
        _exp_exceptions = [None] * num_trials
        try:
            with nogil:
                cimba_run_experiment(
                    <void *>&view[0],
                    num_trials,
                    sizeof(uint64_t),
                    &_py_trial_trampoline,
                )
            results = _exp_results
            exceptions = _exp_exceptions
        finally:
            _exp_trial_fn = None
            _exp_seeds = None
            _exp_results = None
            _exp_exceptions = None

    for exc in exceptions:
        if exc is not None:
            raise exc
    return results


def run_experiment(
    object trial_fn,
    object n=None,
    *,
    object seed=None,
    object seeds=None,
    object backend="process",
    object processes=None,
) -> list:
    """Run independent replications of ``trial_fn``.

    ``trial_fn(index, seed)`` is called once per replication; it typically builds
    and runs a :class:`Simulation` and returns a result. Results are returned in
    a list indexed by replication.

    Parameters
    ----------
    trial_fn:
        Callable ``trial_fn(index: int, seed: int | None) -> Any``.
    n:
        Number of replications. Required unless ``seeds`` is given.
    seed:
        Base seed; per-trial seeds are derived as ``fmix64(seed, index)`` for
        independent streams. Defaults to a hardware seed. Mutually exclusive
        with ``seeds``.
    seeds:
        Explicit per-trial seeds (one per replication); sets ``n`` if omitted.
    backend:
        ``"process"`` (default) runs trials in forked Python worker processes.
        ``"thread"`` runs trials in Cimba's native pthread pool.
    processes:
        Process count for ``backend="process"``. ``None`` delegates to
        :class:`multiprocessing.Pool`.
    """
    if not callable(trial_fn):
        raise TypeError("trial_fn must be callable")

    if backend not in ("process", "thread"):
        raise ValueError("backend must be 'process' or 'thread'")
    if backend == "thread" and processes is not None:
        raise ValueError("processes is only valid with backend='process'")

    cdef list seed_list = _resolve_seeds(n, seed, seeds)
    if backend == "process":
        return _run_process_experiment(trial_fn, seed_list, processes)
    return _run_thread_experiment(trial_fn, seed_list)
