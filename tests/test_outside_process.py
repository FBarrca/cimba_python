"""Hooks and events run outside any process: a verb that would wait fails the trial.

The engine asserts and aborts the interpreter when the dispatcher tries to
wait. These trials must fail with a reason instead, and verbs that complete
without waiting must keep working there.
"""

import pytest

import cimba as cb


class Hold(cb.Model):
    @cb.on_start
    def start(self):
        cb.hold(1.0)


class Suspend(cb.Model):
    @cb.on_end
    def end(self):
        cb.suspend()


class NestedHold(cb.Model):
    @cb.function
    def wait_a_bit(self) -> float:
        cb.hold(1.0)
        return cb.now()

    @cb.event
    def fire(self):
        self.wait_a_bit()

    @cb.process
    def kick(self):
        cb.schedule(self.fire, 1.0)


class EmptyStoreGet(cb.Model):
    items: cb.Store[float]

    @cb.on_start
    def start(self):
        self.items.get()


class FullStorePut(cb.Model):
    items: cb.Store[int] = cb.Store(capacity=1)

    @cb.on_start
    def start(self):
        self.items.put(1)
        self.items.put(2)


class EmptyPriorityGet(cb.Model):
    ranked: cb.PriorityStore[int]

    @cb.on_start
    def start(self):
        self.ranked.get()


class FullPriorityPut(cb.Model):
    ranked: cb.PriorityStore[float] = cb.PriorityStore(capacity=1)

    @cb.on_start
    def start(self):
        self.ranked.put(1.0, 0)
        self.ranked.put(2.0, 0)


class ShortContainerGet(cb.Model):
    bin: cb.Container = cb.Container(initial=2)

    @cb.on_start
    def start(self):
        self.bin.get(3)


class HookAcquire(cb.Model):
    crane: cb.Resource

    @cb.on_start
    def start(self):
        self.crane.acquire()


class EventWait(cb.Model):
    gate: cb.Condition

    @cb.predicate
    def never(self):
        return False

    @cb.event
    def fire(self):
        self.gate.wait_until(self.never)

    @cb.process
    def kick(self):
        cb.schedule(self.fire, 1.0)


@pytest.mark.parametrize(("model", "reason"), [
    (Hold, "hold() cannot wait outside a process"),
    (Suspend, "suspend() cannot wait outside a process"),
    (NestedHold, "hold() cannot wait outside a process"),
    (EmptyStoreGet, "emptystoreget.items: get() would wait for an item outside a process"),
    (FullStorePut, "fullstoreput.items: put() would wait for space outside a process"),
    (EmptyPriorityGet,
     "emptypriorityget.ranked: get() would wait for an item outside a process"),
    (FullPriorityPut,
     "fullpriorityput.ranked: put() would wait for space outside a process"),
    (ShortContainerGet,
     "shortcontainerget.bin: get() would wait for content outside a process"),
    (HookAcquire, "hookacquire.crane: acquire() cannot hold a resource outside a process"),
    (EventWait, "eventwait.gate: wait_until() would wait outside a process"),
])
def test_waiting_outside_a_process_fails_the_trial(model, reason):
    results = cb.Experiment(model()).run(workers=1)
    assert results.failed.all()
    assert results.failure_reasons[0][0] == reason


class NoWait(cb.Model):
    items: cb.Store[float]
    ranked: cb.PriorityStore[int]
    bin: cb.Container = cb.Container(initial=5)
    gate: cb.Condition
    total: cb.Output[float]

    @cb.predicate
    def always(self):
        return True

    @cb.on_start
    def start(self):
        self.items.put(1.0)
        self.ranked.put(4, 0)
        self.bin.get(2)

    @cb.event
    def fire(self):
        self.items.put(self.items.get() + 1.0)
        self.gate.wait_until(self.always)

    @cb.process
    def work(self):
        cb.schedule(self.fire, 1.0)
        cb.hold(2.0)
        self.total = self.items.get() + self.ranked.get() + self.bin.level()


def test_verbs_that_need_not_wait_still_work_outside_a_process():
    model = NoWait()
    results = cb.Experiment(model).run(workers=1)
    assert not results.failed.any()
    assert results[model].total.values.tolist() == [[2.0 + 4 + 3]]
