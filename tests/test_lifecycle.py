import subprocess
import sys
import textwrap

import numpy as np
import pytest

import cimba.sim as sim


class LifecycleModel(sim.Model):
    completed: sim.Output
    samples: sim.Output
    queue: sim.Queue = 4
    resource: sim.Resource
    pool: sim.Pool = 2
    store: sim.Store = 4
    dataset: sim.Dataset
    condition: sim.Condition
    priority_queues: sim.PQueues = sim.count(1)

    @sim.process
    def exercise_entities(self):
        self.queue.put(1)
        self.resource.acquire()
        self.pool.acquire(1)
        self.store.put(17)
        self.dataset.add(3.0)
        self.priority_queues[0].put(23, 5)
        self.completed = 1.0
        # Deliberately leave a live process, held resources, and populated
        # containers for the end-of-trial lifecycle to clean up.
        sim.suspend()

    @sim.collect
    def collect(self):
        self.samples = float(self.dataset.count())


class EarlyExitModel(sim.Model):
    completed: sim.Output

    @sim.process(spawnable=True)
    def child(self):
        sim.suspend()

    @sim.process
    def clear_events_while_processes_are_live(self):
        sim.spawn(self.child, self)
        # Let the spawned process start and suspend before removing every
        # remaining event, including the normal end-of-trial stop event.
        sim.hold(0.0)
        self.completed = 1.0
        sim.clear_events()
        sim.suspend()


def test_compiled_lifecycle_uses_ownership_appropriate_cleanup(monkeypatch):
    import multiprocessing

    def no_fork(_method):
        raise ValueError("inspect lifecycle callbacks in the parent process")

    monkeypatch.setenv("CIMBA_CACHE", "0")
    monkeypatch.setattr(multiprocessing, "get_context", no_fork)
    model = LifecycleModel()
    model.experiment(replications=1, duration=1.0, warmup=0.0, seed=7)

    assert model._compiled is not None
    (
        _recording,
        initialize_trial,
        initialize_entities,
        initialize_processes,
        teardown_trial,
        stop_trial,
        cleanup_processes,
        _collect,
    ) = model._compiled["events"]

    trial_init_ir = initialize_trial.inspect_llvm()
    entity_init_ir = initialize_entities.inspect_llvm()
    process_init_ir = initialize_processes.inspect_llvm()
    teardown_ir = teardown_trial.inspect_llvm()
    process_stop_ir = stop_trial.inspect_llvm()
    process_cleanup_ir = cleanup_processes.inspect_llvm()

    for family in (
        "buffer",
        "resource",
        "resourcepool",
        "objectqueue",
        "dataset",
        "condition",
        "priorityqueue",
    ):
        assert f"cmb_{family}_create" in entity_init_ir
        assert f"cmb_{family}_initialize" in entity_init_ir
        assert f"cmb_{family}_destroy" in teardown_ir
        # RC2 requires explicit termination before destroying every entity.
        assert f"cmb_{family}_terminate" in teardown_ir

    assert "cmb_event_queue_initialize" in trial_init_ir
    assert "cmb_random_initialize" in trial_init_ir
    assert "cmb_event_queue_terminate" in teardown_ir
    assert "cmb_random_terminate" in teardown_ir
    assert "cmb_process_create" in process_init_ir
    assert "cmb_process_initialize" in process_init_ir
    assert "cmb_process_stop" in process_stop_ir
    assert "cmb_process_terminate" in process_cleanup_ir
    assert "cmb_process_destroy" in process_cleanup_ir
    assert "cpy_spawned_stop_all" in process_stop_ir


@pytest.mark.parametrize("duration", [1.0, None], ids=["timed", "idle"])
def test_lifecycle_cleanup_supports_reused_workers_and_experiment_reruns(duration):
    model = LifecycleModel()
    experiment = model.experiment(
        replications=32,
        duration=duration,
        warmup=0.0,
        seed=11,
    )

    for _ in range(3):
        assert experiment.run() == 0
        np.testing.assert_array_equal(experiment["completed"], 1.0)
        np.testing.assert_array_equal(experiment["samples"], 1.0)


@pytest.mark.parametrize("duration", [10.0, None], ids=["timed", "idle"])
def test_early_event_queue_exit_stops_static_and_spawned_processes(duration):
    model = EarlyExitModel()
    experiment = model.experiment(
        replications=32,
        duration=duration,
        warmup=0.0,
        seed=13,
    )

    for _ in range(3):
        assert experiment.run() == 0
        np.testing.assert_array_equal(experiment["completed"], 1.0)


def test_abandoned_trials_clear_spawned_registry_before_worker_reuse():
    # Keep native assertions/invalid-pointer failures isolated from pytest.
    # The trial callback is entirely native: longjmp must never cross Python.
    script = textwrap.dedent('''
        import ctypes
        import numpy as np
        from numba import carray, cfunc, types
        from cimba import _bindings as b, _cimba_native

        native = ctypes.CDLL(_cimba_native.__file__)
        native.cimba_threads_use.argtypes = [ctypes.c_uint32]
        native.cimba_threads_use.restype = ctypes.c_uint32
        native.cimba_threads_use(1)
        native.cimba_run.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                    ctypes.c_size_t, ctypes.c_void_p]
        native.cimba_run.restype = ctypes.c_uint64
        native.cpy_process_sizeof.restype = ctypes.c_uint64
        size = native.cpy_process_sizeof() + 64
        abandon = types.ExternalFunction("cimba_trial_abandon", types.void())
        name = b.cstring("lifecycle recovery")

        @cfunc(types.intp(types.intp, types.intp))
        def body(process, context):
            # Abandon from an active coroutine, after earlier processes have
            # suspended. This exercises recovery across a stack switch.
            if context == 1:
                abandon()
            b.process_yield()
            return 0

        body_address = body.address

        @cfunc(types.void(types.CPointer(types.int64)))
        def trial(ptr):
            fields = carray(ptr, 2)
            b.event_queue_initialize(0.0)
            # Force registry growth and exercise extended process allocations.
            for i in range(20):
                process = b.process_create_sized(size)
                should_abandon = int(i == 19 and fields[0] % 2 == 0)
                b.process_initialize(process, name, body_address, should_abandon, 0)
                b.spawned_register(process)
                b.process_start(process)
            b.event_queue_execute()
            b.spawned_stop_all()
            b.spawned_reclaim()
            b.event_queue_terminate()
            fields[1] = 1

        for _ in range(3):
            trials = np.zeros((32, 2), dtype=np.int64)
            trials[:, 0] = np.arange(32)
            failures = native.cimba_run(trials.ctypes.data, 32,
                                       trials.strides[0], trial.address)
            assert failures == 16, failures
            np.testing.assert_array_equal(trials[:, 1], np.arange(32) % 2)
    ''')
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class FiniteIdleModel(sim.Model):
    store: sim.Store
    dataset: sim.Dataset
    completed: sim.Output
    finished_at: sim.Output
    samples: sim.Output
    recorded: sim.Output

    @sim.process
    def producer(self):
        sim.spawn(self.consumer, self)
        for index in range(3):
            sim.hold(1.0)
            self.store.put(index + 1)

    @sim.process(spawnable=True)
    def consumer(self):
        self.completed = 0.0
        while True:
            value = self.store.take()
            self.dataset.add(float(value))
            self.completed += 1.0

    @sim.collect
    def collect(self):
        self.finished_at = sim.now()
        self.samples = float(self.dataset.count())
        self.recorded = float(self.store.history().count())


class EmptyIdleModel(sim.Model):
    finished_at: sim.Output

    @sim.process
    def no_work(self):
        pass

    @sim.collect
    def collect(self):
        self.finished_at = sim.now()


def test_idle_trials_drain_work_and_collect_before_cleanup():
    experiment = FiniteIdleModel().experiment(
        duration=None, warmup=0.0, start_time=5.0, replications=32, seed=41)
    for _ in range(3):
        assert experiment.run() == 0
        np.testing.assert_array_equal(experiment['completed'], 3.0)
        np.testing.assert_array_equal(experiment['finished_at'], 8.0)
        # There is no automatic warmup reset; explicit dataset sampling stays.
        np.testing.assert_array_equal(experiment['samples'], 3.0)
        np.testing.assert_array_equal(experiment['recorded'], 0.0)


def test_idle_trial_without_work_collects_at_start_time():
    experiment = EmptyIdleModel().experiment(
        duration=None, warmup=0.0, start_time=7.0, replications=2)
    assert experiment.run() == 0
    np.testing.assert_array_equal(experiment['finished_at'], 7.0)


def test_idle_mode_rejects_recording_window_and_cooldown():
    model = EmptyIdleModel()
    for kwargs in ({}, {'warmup': 1.0}, {'warmup': 0.0, 'cooldown': 1.0}):
        with pytest.raises(ValueError, match='duration=None requires'):
            model.experiment(duration=None, **kwargs)
    for duration in (float('inf'), float('-inf'), float('nan')):
        with pytest.raises(ValueError, match='duration must be finite'):
            model.experiment(duration=duration, warmup=0.0)
