"""Condition(observe=...): a condition signalled by every change to the
entities it observes, with no manual signal(). Exact timestamps throughout."""

from __future__ import annotations

import pytest

import cimba as cb
from cimba.schema import ModelDefinitionError


def run(model, **kwargs):
    result = cb.Experiment(model, **kwargs).run(workers=1)
    assert not result.failed.any(), result.failure_reasons
    return result[model]


def value(output):
    return output.values.tolist()[0][0]


class Office(cb.Model):
    """Holders keep the clerk until t=10 and the senior until t=5. Either
    waits for any free role, Clerk for the clerk only (first in line) and
    Senior for the senior only: each wakes when its own predicate holds."""
    clerks: cb.Resource = cb.Resource(capacity=1)
    seniors: cb.Resource = cb.Resource(capacity=1)
    staffed: cb.Condition = cb.Condition(observe=(clerks, seniors))
    either_at: cb.Output[float]
    clerk_at: cb.Output[float]
    senior_at: cb.Output[float]

    @cb.process(priority=9)
    def holders(self):
        self.clerks.acquire()
        self.seniors.acquire()
        cb.hold(5.0)
        self.seniors.release()
        cb.hold(5.0)
        self.clerks.release()

    @cb.predicate
    def any_free(self) -> bool:
        return self.clerks.available() > 0 or self.seniors.available() > 0

    @cb.predicate
    def clerk_free(self) -> bool:
        return self.clerks.available() > 0

    @cb.predicate
    def senior_free(self) -> bool:
        return self.seniors.available() > 0

    @cb.process
    def clerk(self):
        cb.hold(1.0)
        self.staffed.wait_until(self.clerk_free)
        self.clerk_at = cb.now()

    @cb.process
    def senior(self):
        cb.hold(2.0)
        self.staffed.wait_until(self.senior_free)
        self.senior_at = cb.now()

    @cb.process
    def either(self):
        cb.hold(3.0)
        self.staffed.wait_until(self.any_free)
        self.either_at = cb.now()


def test_releases_signal_an_observing_condition():
    out = run(Office())
    assert value(out.senior_at) == 5.0   # not held up by Clerk, first in line
    assert value(out.either_at) == 5.0
    assert value(out.clerk_at) == 10.0


class Job(cb.Model):
    size: float = 1.0


class Sizes(cb.Model):
    """Waits on a Store, a PriorityStore, a Container, a FractionalContainer
    and a Resource's capacity, each signalled only by the entity itself."""
    jobs: cb.Store[Job] = cb.Store()
    urgent: cb.PriorityStore[Job] = cb.PriorityStore()
    parts: cb.Container = cb.Container(initial=5)
    flour: cb.FractionalContainer = cb.FractionalContainer(1.0)
    staff: cb.Resource = cb.Resource(capacity=1)
    changed: cb.Condition = cb.Condition(observe=jobs)
    jobs_at: cb.Output[float]
    urgent_at: cb.Output[float]
    parts_at: cb.Output[float]
    flour_at: cb.Output[float]
    staff_at: cb.Output[float]

    def __init__(self):
        self.changed.observe(self.urgent, self.parts, self.flour, self.staff)

    @cb.predicate
    def two_jobs(self) -> bool:
        return self.jobs.length() >= 2

    @cb.predicate
    def has_urgent(self) -> bool:
        return self.urgent.length() > 0

    @cb.predicate
    def parts_low(self) -> bool:
        return self.parts.level() <= 2

    @cb.predicate
    def flour_full(self) -> bool:
        return self.flour.level() >= 2.5

    @cb.predicate
    def staff_two(self) -> bool:
        return self.staff.capacity() >= 2

    @cb.process
    def producer(self):
        cb.hold(1.0)
        self.jobs.put(cb.spawn(Job))
        cb.hold(2.0)
        self.jobs.put(cb.spawn(Job))    # t=3
        cb.hold(1.0)
        self.urgent.put(cb.spawn(Job), 3)  # t=4
        cb.hold(1.0)
        self.parts.get(3)           # t=5
        cb.hold(1.0)
        self.flour.put(1.5)         # t=6
        cb.hold(1.0)
        self.staff.set_capacity(2)  # t=7

    @cb.process
    def jobs_waiter(self):
        self.changed.wait_until(self.two_jobs)
        self.jobs_at = cb.now()

    @cb.process
    def urgent_waiter(self):
        self.changed.wait_until(self.has_urgent)
        self.urgent_at = cb.now()

    @cb.process
    def parts_waiter(self):
        self.changed.wait_until(self.parts_low)
        self.parts_at = cb.now()

    @cb.process
    def flour_waiter(self):
        self.changed.wait_until(self.flour_full)
        self.flour_at = cb.now()

    @cb.process
    def staff_waiter(self):
        self.changed.wait_until(self.staff_two)
        self.staff_at = cb.now()


def test_stores_and_containers_signal_an_observing_condition():
    out = run(Sizes())
    assert value(out.jobs_at) == 3.0
    assert value(out.urgent_at) == 4.0
    assert value(out.parts_at) == 5.0
    assert value(out.flour_at) == 6.0
    assert value(out.staff_at) == 7.0


class Branch(cb.Model):
    """One counter; its own condition, declared with the class, observes it."""
    counter: cb.Resource = cb.Resource(capacity=1)
    busy_until: cb.Param[float] = 0.0
    local: cb.Condition = cb.Condition(observe=counter)
    free_at: cb.Output[float]

    @cb.process(priority=9)
    def busy(self):
        self.counter.acquire()
        cb.hold(self.busy_until)
        self.counter.release()

    @cb.predicate
    def free(self) -> bool:
        return self.counter.available() > 0

    @cb.process
    def watch(self):
        cb.hold(0.5)
        self.local.wait_until(self.free)
        self.free_at = cb.now()


class Bank(cb.Model):
    """The bank's condition observes both branches' counters. Each branch's
    own condition observes its own counter only."""
    north: Branch
    south: Branch
    either_at: cb.Output[float]

    def __init__(self):
        self.north = Branch()
        self.north.busy_until = 4.0
        self.south = Branch()
        self.south.busy_until = 9.0
        self.any_counter = cb.Condition(observe=(self.north.counter,
                                                 self.south.counter))

    any_counter: cb.Condition

    @cb.predicate
    def counter_free(self) -> bool:
        return self.north.counter.available() > 0 or \
            self.south.counter.available() > 0

    @cb.process
    def watch(self):
        cb.hold(1.0)
        self.any_counter.wait_until(self.counter_free)
        self.either_at = cb.now()


def test_conditions_observe_entities_across_child_models():
    bank = Bank()
    results = cb.Experiment(bank).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[bank].either_at.values.tolist() == [[4.0]]
    assert results[bank.north].free_at.values.tolist() == [[4.0]]
    assert results[bank.south].free_at.values.tolist() == [[9.0]]


class Wing(cb.Model):
    """Observes its owner's pool, bound when it is attached."""
    near: cb.Condition = cb.Condition()
    seen_at: cb.Output[float]
    pool: cb.Ref[Pool]

    def attached(self, owner, field):
        self.pool = owner
        self.near.observe(owner.desks)

    @cb.predicate
    def desk_free(self) -> bool:
        return self.pool.desks.available() > 0

    @cb.process
    def watch(self):
        self.near.wait_until(self.desk_free)
        self.seen_at = cb.now()


class Pool(cb.Model):
    desks: cb.Resource = cb.Resource(capacity=1)
    wings: list[Wing]

    def __init__(self):
        self.wings = [Wing(), Wing()]

    @cb.process(priority=9)
    def busy(self):
        self.desks.acquire()
        cb.hold(6.0)
        self.desks.release()


def test_a_child_observes_its_owners_entity():
    pool = Pool()
    results = cb.Experiment(pool).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    for wing in pool.wings:
        assert results[wing].seen_at.values.tolist() == [[6.0]]


def test_only_resources_stores_and_containers_can_be_observed():
    with pytest.raises(TypeError, match="can observe a Resource"):
        cb.Condition(observe=cb.Dataset())
    with pytest.raises(TypeError, match="can observe a Resource"):
        cb.Condition().observe(cb.Condition())


class Stray(cb.Model):
    watched: cb.Condition = cb.Condition(observe=cb.Resource(capacity=1))

    @cb.process
    def idle(self):
        cb.hold(1.0)


def test_observed_entities_must_be_in_the_model_tree():
    with pytest.raises(ModelDefinitionError, match="not an entity of this model tree"):
        cb.Experiment(Stray()).run(workers=1)


class Crowd(cb.Model):
    """Observing conditions with waiters when the trial is abandoned by
    releasing more than is held."""
    machine: cb.Resource = cb.Resource(capacity=2)
    queue: cb.Store[Job] = cb.Store()
    gate: cb.Condition = cb.Condition(observe=(machine, queue))
    bad: cb.Param[int] = 1

    @cb.predicate
    def never(self) -> bool:
        return self.queue.length() > 100

    @cb.process(copies=3)
    def waiter(self):
        self.gate.wait_until(self.never)

    @cb.process
    def user(self):
        self.machine.acquire()
        self.queue.put(cb.spawn(Job))
        cb.hold(2.0)
        self.machine.release()

    @cb.process
    def culprit(self):
        cb.hold(3.0)
        if self.bad:
            self.machine.release(1)


def test_abandoned_trials_with_observers_leak_nothing():
    model = Crowd()
    results = cb.Experiment(model, replications=200).run(workers=1)
    assert results.failed.all()
    assert "release(1) exceeds the units held" in results.failure_reasons[0][0]
    model.bad = 0
    assert not cb.Experiment(model, replications=50,
                             window=cb.Window(duration=10)).run(workers=1).failed.any()


def test_class_default_entities_are_copied_per_instance():
    first, second = Branch(), Branch()
    assert first.counter is not second.counter
    assert first.local.observed == (first.counter,)
    assert second.local.observed == (second.counter,)
    assert Branch.local.observed == (Branch.counter,)


def test_process_graph_draws_observations():
    from cimba.diagrams import process_graph

    edges = {(e.source, e.label, e.target, e.style)
             for e in process_graph(Bank()).edges}
    assert ("bank.north.counter", "observe", "bank.any_counter", "dotted") in edges
    assert ("bank.south.counter", "observe", "bank.any_counter", "dotted") in edges
    assert ("bank.north.counter", "observe", "bank.north.local", "dotted") in edges
