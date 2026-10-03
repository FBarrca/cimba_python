"""A static record can finish its work without ending the surrounding trial."""

import pytest

import cimba as cb
from cimba.compiler import ModelCompileError


class Job(cb.Model):
    completed: cb.State[bool] = False
    ticks: cb.State[int] = 0
    after_release: cb.State[bool] = False
    reported: cb.Output[bool]
    total: cb.Output[int]

    @cb.process
    def finish(self):
        cb.hold(0.5)
        self.completed = True
        cb.release(self)
        self.after_release = True

    @cb.process(copies=2)
    def work(self):
        while True:
            cb.hold(1.0)
            self.ticks += 1

    @cb.on_end
    def report(self):
        self.reported = self.completed and not self.after_release
        self.total = self.ticks


class Owner(cb.Model):
    job: Job
    seen: cb.Output[bool]
    continued: cb.Output[bool]

    def __init__(self):
        self.job = Job()

    @cb.process
    def check(self):
        cb.hold(2.0)
        self.seen = self.job.completed and not cb.is_dynamic(self.job)
        self.continued = True


@pytest.mark.parametrize("workers", [1, 2])
def test_static_self_release_stops_all_processes_and_preserves_record_and_end_hooks(workers):
    model = Owner()
    experiment = cb.Experiment(model, replications=3)
    for _ in range(2):
        result = experiment.run(workers=workers, on_failure="raise")
        assert result[model].seen.values.all()
        assert result[model].continued.values.all()
        assert result[model.job].reported.values.all()
        assert not result[model.job].total.values.any()


@pytest.mark.parametrize("on_start", [False, True])
def test_static_external_release_before_process_start(on_start):
    class IdleJob(cb.Model):
        started: cb.Output[int]
        end_hook: cb.Output[bool]

        @cb.process(copies=2)
        def work(self):
            self.started += 1

        @cb.on_end
        def report(self):
            self.end_hook = True

    class Parent(cb.Model):
        job: IdleJob
        continued: cb.Output[bool]

        @cb.on_start
        def start(self):
            if on_start:
                cb.release(self.job)

        @cb.process(priority=1)
        def close(self):
            if not on_start:
                cb.release(self.job)
            cb.hold(1.0)
            self.continued = True

    model = Parent()
    model.job = IdleJob()
    result = cb.Experiment(model).run(workers=1, on_failure="raise")
    assert result[model.job].started[0, 0] == 0
    assert result[model.job].end_hook[0, 0] == 1
    assert result[model].continued[0, 0] == 1


def test_tree_release_does_not_stop_children():
    class Child(cb.Model):
        ran: cb.Output[bool]

        @cb.process
        def work(self):
            cb.hold(1.0)
            self.ran = True

    class Parent(cb.Model):
        child: Child

        @cb.process
        def finish(self):
            cb.release(self)

    model = Parent()
    model.child = Child()
    result = cb.Experiment(model).run(workers=1, on_failure="raise")
    assert result[model.child].ran[0, 0] == 1


def test_repeated_tree_release_fails_and_worker_can_recover():
    class Parent(cb.Model):
        child: cb.Model
        twice: cb.Param[bool] = False
        done: cb.Output[bool]

        @cb.process
        def close(self):
            cb.release(self.child)
            if self.twice:
                cb.release(self.child)
            self.done = True

    model = Parent()
    model.child = cb.Model()
    model.twice = cb.sweep(True, False)
    result = cb.Experiment(model, replications=2).run(workers=1)
    assert result.failed.tolist() == [[True, True], [False, False]]
    assert result[model].done.values[1].all()


def test_release_rejects_entity_handles():
    class Invalid(cb.Model):
        gate: cb.Condition

        @cb.process
        def close(self):
            cb.release(self.gate)

    with pytest.raises(ModelCompileError, match="release"):
        cb.Experiment(Invalid()).run(workers=1)
