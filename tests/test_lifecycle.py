import numpy as np

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
        _stop_trial,
        cleanup_processes,
        _collect,
    ) = model._compiled["events"]

    trial_init_ir = initialize_trial.inspect_llvm()
    entity_init_ir = initialize_entities.inspect_llvm()
    process_init_ir = initialize_processes.inspect_llvm()
    teardown_ir = teardown_trial.inspect_llvm()
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
    assert "cmb_process_stop" in process_cleanup_ir
    assert "cmb_process_terminate" in process_cleanup_ir
    assert "cmb_process_destroy" in process_cleanup_ir
    assert "cpy_spawned_stop_all" in process_cleanup_ir


def test_lifecycle_cleanup_supports_reused_workers_and_experiment_reruns():
    model = LifecycleModel()
    experiment = model.experiment(
        replications=32,
        duration=1.0,
        warmup=0.0,
        seed=11,
    )

    for _ in range(3):
        assert experiment.run() == 0
        np.testing.assert_array_equal(experiment["completed"], 1.0)
        np.testing.assert_array_equal(experiment["samples"], 1.0)


def test_early_event_queue_exit_stops_static_and_spawned_processes():
    model = EarlyExitModel()
    experiment = model.experiment(
        replications=32,
        duration=10.0,
        warmup=0.0,
        seed=13,
    )

    for _ in range(3):
        assert experiment.run() == 0
        np.testing.assert_array_equal(experiment["completed"], 1.0)


class FiniteIdleModel(sim.Model):
    store: sim.Store
    dataset: sim.Dataset
    completed: sim.Output
    finished_at: sim.Output
    samples: sim.Output

    @sim.process
    def producer(self):
        for index in range(3):
            sim.hold(1.0)
            self.store.put(index + 1)

    @sim.process
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


def test_idle_trial_without_work_collects_at_start_time():
    experiment = EmptyIdleModel().experiment(
        duration=None, warmup=0.0, start_time=7.0, replications=2)
    assert experiment.run() == 0
    np.testing.assert_array_equal(experiment['finished_at'], 7.0)


def test_idle_trials_clean_up_all_entity_types_and_suspended_processes():
    experiment = LifecycleModel().experiment(
        duration=None, warmup=0.0, replications=32, seed=19)
    for _ in range(3):
        assert experiment.run() == 0
        np.testing.assert_array_equal(experiment['completed'], 1.0)
        np.testing.assert_array_equal(experiment['samples'], 1.0)


def test_idle_trials_clean_up_spawned_processes():
    experiment = EarlyExitModel().experiment(
        duration=None, warmup=0.0, replications=32, seed=23)
    for _ in range(3):
        assert experiment.run() == 0
        np.testing.assert_array_equal(experiment['completed'], 1.0)


def test_idle_mode_rejects_recording_window_and_cooldown():
    import pytest

    model = EmptyIdleModel()
    for kwargs in ({}, {'warmup': 1.0}, {'warmup': 0.0, 'cooldown': 1.0}):
        with pytest.raises(ValueError, match='duration=None requires'):
            model.experiment(duration=None, **kwargs)
    for duration in (float('inf'), float('-inf'), float('nan')):
        with pytest.raises(ValueError, match='duration must be finite'):
            model.experiment(duration=duration, warmup=0.0)
