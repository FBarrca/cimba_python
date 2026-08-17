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
        # These are heap-created base entities. Their destroy functions own
        # termination, unlike cmb_process_destroy, so a direct terminate call
        # here would be a double teardown.
        assert f"cmb_{family}_terminate" not in teardown_ir

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
