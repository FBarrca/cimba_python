import os
import sys
import threading

import pytest

import cimba


_GIL_ENABLED = getattr(sys, "_is_gil_enabled", lambda: True)()

requires_freethreading = pytest.mark.skipif(
    _GIL_ENABLED,
    reason="real parallelism requires a free-threaded interpreter (python3.13t)",
)


def _mm1_mean_queue(i, seed):
    """A small M/M/1 replication: build + run a Simulation, return a metric."""
    with cimba.Simulation(seed=seed) as sim:
        queue = cimba.Buffer("Queue")
        queue.start_recording()

        def arrival(q):
            while True:
                cimba.hold(cimba.random.exponential(1.0 / 0.75))
                q.put(1)

        def service(q):
            while True:
                q.get(1)
                cimba.hold(cimba.random.exponential(1.0))

        cimba.Process("Arrival", arrival, queue).start()
        cimba.Process("Service", service, queue).start()
        sim.stop_at(2000.0)
        sim.execute()
        queue.stop_recording()
        return (i, seed, queue.history().summary().mean)


# --- correctness (runs on any interpreter) --------------------------------


def test_results_are_indexed_by_replication():
    rows = cimba.run_experiment(_mm1_mean_queue, 8, seed=42, processes=2)
    assert [row[0] for row in rows] == list(range(8))


def test_same_seed_is_deterministic():
    a = cimba.run_experiment(_mm1_mean_queue, 1, seed=123, processes=1)
    b = cimba.run_experiment(_mm1_mean_queue, 1, seed=123, processes=1)
    assert a[0][2] == b[0][2]


def test_derived_seeds_are_independent_streams():
    rows = cimba.run_experiment(_mm1_mean_queue, 8, seed=42, processes=2)
    seeds = [row[1] for row in rows]
    means = [row[2] for row in rows]
    assert len(set(seeds)) == 8
    assert len(set(means)) == 8


def test_explicit_seeds_set_count_and_are_passed_through():
    rows = cimba.run_experiment(_mm1_mean_queue, seeds=[1, 2, 3], processes=2)
    assert len(rows) == 3
    assert [row[1] for row in rows] == [1, 2, 3]


def test_trial_exception_propagates_after_run():
    def boom(i, seed):
        raise ValueError(f"trial {i} failed")

    with pytest.raises(ValueError, match="failed"):
        cimba.run_experiment(boom, 4, seed=1, processes=1)


def test_zero_replications_is_empty():
    assert cimba.run_experiment(_mm1_mean_queue, 0) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # neither n nor seeds
        {"n": 2, "seed": 1, "seeds": [1, 2]},  # seed and seeds together
        {"n": 3, "seeds": [1, 2]},  # n disagrees with len(seeds)
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        cimba.run_experiment(_mm1_mean_queue, **kwargs)


def test_non_callable_trial_fn_raises():
    with pytest.raises(TypeError):
        cimba.run_experiment(object(), 1)


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        cimba.run_experiment(_mm1_mean_queue, 1, backend="bogus")


@pytest.mark.parametrize("processes", [0, -1])
def test_invalid_process_count_raises(processes):
    with pytest.raises(ValueError, match="processes"):
        cimba.run_experiment(_mm1_mean_queue, 1, processes=processes)


def test_processes_rejected_for_thread_backend():
    with pytest.raises(ValueError, match="processes"):
        cimba.run_experiment(_mm1_mean_queue, 1, backend="thread", processes=1)


def test_default_process_backend_runs_in_child_process():
    parent_pid = os.getpid()

    def trial(i, seed):
        return os.getpid()

    child_pids = cimba.run_experiment(trial, 1, processes=1)
    assert len(child_pids) == 1
    assert child_pids[0] != parent_pid


def test_local_trial_function_works_with_process_backend_fork():
    offset = 10

    def trial(i, seed):
        return i + offset

    assert cimba.run_experiment(trial, 3, processes=1) == [10, 11, 12]


@pytest.mark.parametrize("processes", [1, 2])
def test_process_backend_accepts_process_count(processes):
    assert cimba.run_experiment(lambda i, seed: i, 3, processes=processes) == [0, 1, 2]


@pytest.mark.skipif(not _GIL_ENABLED, reason="serial warning only on a GIL build")
def test_warns_when_serial_under_gil():
    with pytest.warns(RuntimeWarning, match="serially"):
        cimba.run_experiment(lambda i, seed: i, 2, seed=1, backend="thread")


def test_thread_backend_can_return_native_summary_object():
    def trial(i, seed):
        summary = cimba.DataSummary()
        summary.add(1.0)
        summary.add(3.0)
        return summary

    (summary,) = cimba.run_experiment(trial, 1, backend="thread")
    assert summary.mean == 2.0


# --- real parallelism (free-threaded interpreter only) --------------------


@requires_freethreading
@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs >= 2 logical cores")
def test_trials_run_concurrently_on_distinct_threads():
    parties = 2
    barrier = threading.Barrier(parties, timeout=30)

    def trial(i, seed):
        # Only completes if >= `parties` trials are executing simultaneously,
        # which is impossible under a serializing interpreter.
        barrier.wait()
        return threading.get_ident()

    tids = cimba.run_experiment(trial, parties, backend="thread")
    assert len(set(tids)) == parties
