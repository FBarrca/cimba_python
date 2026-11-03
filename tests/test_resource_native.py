"""Resource's own native implementation (resource.c): queue places, cleanup
of waits and holdings when processes are interrupted, preempted, stopped or
abandoned, and priority changes of holders. Exact timestamps throughout."""

import cimba as cb

PREEMPTED = -1


def run(model, **kwargs):
    result = cb.Experiment(model, **kwargs).run(workers=1)
    assert not result.failed.any(), result.failure_reasons
    return result[model]


def value(output):
    return output.values.tolist()[0][0]


class PartialFifo(cb.Model):
    """H1 and H2 hold the two units until t=10 and t=20. First asks for 2 at
    t=1: it takes H1's unit at t=10 and waits again. Second asks for 1 at
    t=2. At t=20 First is still ahead of Second, so it completes then, and
    Second waits until First releases at t=25."""
    machine: cb.Resource = cb.Resource(capacity=2)
    request: cb.Param[int] = -1
    first_at: cb.Output[float]
    second_at: cb.Output[float]
    free_at_15: cb.Output[int]

    @cb.process
    def h1(self):
        self.machine.acquire()
        cb.hold(10.0)
        self.machine.release()

    @cb.process
    def h2(self):
        self.machine.acquire()
        cb.hold(20.0)
        self.machine.release()

    @cb.process
    def first(self):
        cb.hold(1.0)
        if self.request < 0:
            self.machine.acquire(2)
        else:
            self.machine.acquire(2, priority=self.request)
        self.first_at = cb.now()
        cb.hold(5.0)
        self.machine.release(2)

    @cb.process
    def second(self):
        cb.hold(2.0)
        if self.request < 0:
            self.machine.acquire()
        else:
            self.machine.acquire(1, priority=self.request)
        self.second_at = cb.now()
        self.machine.release()

    @cb.process
    def probe(self):
        cb.hold(15.0)
        self.free_at_15 = self.machine.available()  # First has taken H1's unit


def test_partially_served_request_keeps_its_place():
    out = run(PartialFifo())
    assert value(out.free_at_15) == 0
    assert value(out.first_at) == 20.0
    assert value(out.second_at) == 25.0


def test_partially_served_ranked_request_keeps_its_place():
    model = PartialFifo()
    model.request = 3
    out = run(model)
    assert value(out.first_at) == 20.0
    assert value(out.second_at) == 25.0


class StaleGrant(cb.Model):
    """Holder (priority 5) releases at t=10, granting First (waiting since
    t=1). Before First's wakeup runs, Cat (priority 3) interrupts it. First
    returns the signal holding nothing, and the unit goes to Second at t=10:
    the stale grant must not resume First later."""
    machine: cb.Resource = cb.Resource(capacity=1)
    first_pointer: int = 0
    first_signal: cb.Output[int]
    first_held: cb.Output[int]
    first_back: cb.Output[float]
    second_at: cb.Output[float]

    @cb.process(priority=5)
    def holder(self):
        self.machine.acquire()
        cb.hold(10.0)
        self.machine.release()

    @cb.process
    def first(self):
        self.first_pointer = cb.this_process().pointer
        cb.hold(1.0)
        self.first_signal = self.machine.acquire()
        self.first_held = self.machine.held(cb.this_process())
        self.first_back = cb.now()
        cb.hold(100.0)

    @cb.process
    def second(self):
        cb.hold(2.0)
        self.machine.acquire()
        self.second_at = cb.now()
        cb.hold(1.0)
        self.machine.release()

    @cb.process(priority=3)
    def cat(self):
        cb.hold(10.0)
        cb.Process(self.first_pointer).interrupt(7, 4)


def test_interrupt_after_grant_passes_the_unit_on():
    out = run(StaleGrant())
    assert value(out.first_signal) == 7
    assert value(out.first_held) == 0
    assert value(out.first_back) == 10.0
    assert value(out.second_at) == 10.0


class PreemptedElsewhere(cb.Model):
    """Victim holds A and waits in B behind Blocker. Thief (higher priority)
    preempts A at t=3: Victim's wait in B ends with -1 at t=3, holding
    nothing anywhere, and Later, who queued in B after Victim, is served when
    Blocker releases B at t=10."""
    a: cb.Resource = cb.Resource(capacity=1)
    b: cb.Resource = cb.Resource(capacity=1)
    victim_signal: cb.Output[int]
    victim_at: cb.Output[float]
    victim_held: cb.Output[int]
    later_at: cb.Output[float]
    thief_at: cb.Output[float]

    @cb.process
    def blocker(self):
        self.b.acquire()
        cb.hold(10.0)
        self.b.release()

    @cb.process
    def victim(self):
        cb.hold(1.0)
        self.a.acquire()
        self.victim_signal = self.b.acquire()
        self.victim_at = cb.now()
        me = cb.this_process()
        self.victim_held = self.a.held(me) + self.b.held(me)

    @cb.process
    def later(self):
        cb.hold(2.0)
        self.b.acquire()
        self.later_at = cb.now()
        self.b.release()

    @cb.process(priority=5)
    def thief(self):
        cb.hold(3.0)
        self.a.preempt()
        self.thief_at = cb.now()
        self.a.release()


def test_preemption_cancels_a_wait_in_another_resource():
    out = run(PreemptedElsewhere())
    assert value(out.victim_signal) == PREEMPTED
    assert value(out.victim_at) == 3.0
    assert value(out.victim_held) == 0
    assert value(out.thief_at) == 3.0
    assert value(out.later_at) == 10.0


class HolderReprioritized(cb.Model):
    """Two holders at process priority 0, one tracking and one at an explicit
    0. At t=2 both raise their process priority to 10. Thief (priority 5)
    preempts one unit at t=3: only the explicit holding is still below it."""
    machine: cb.Resource = cb.Resource(capacity=2)
    tracking_signal: cb.Output[int]
    fixed_signal: cb.Output[int]

    @cb.process
    def tracking(self):
        self.machine.acquire()
        cb.this_process().priority_set(10)
        self.tracking_signal = cb.hold(10.0)

    @cb.process
    def fixed(self):
        self.machine.acquire(1, priority=0)
        cb.this_process().priority_set(10)
        self.fixed_signal = cb.hold(10.0)

    @cb.process(priority=5)
    def thief(self):
        cb.hold(3.0)
        self.machine.preempt()


def test_priority_set_reranks_tracking_holdings_only():
    out = run(HolderReprioritized())
    assert value(out.tracking_signal) == 0
    assert value(out.fixed_signal) == PREEMPTED


class SelfSpared(cb.Model):
    """Other holds one of two units from t=0, Self takes the other at t=1 and
    closes the resource with "preempt": Other is preempted, Self keeps its
    unit, which it releases at t=2."""
    machine: cb.Resource = cb.Resource(capacity=2)
    other_signal: cb.Output[int]
    other_at: cb.Output[float]
    self_held: cb.Output[int]
    in_use_after: cb.Output[int]

    @cb.process
    def other(self):
        self.machine.acquire()
        self.other_signal = cb.hold(100.0)
        self.other_at = cb.now()

    @cb.process
    def spared(self):
        cb.hold(1.0)
        self.machine.acquire()
        self.machine.set_capacity(0, mode="preempt")
        self.self_held = self.machine.held(cb.this_process())
        cb.hold(1.0)
        self.machine.release()
        self.in_use_after = self.machine.held(cb.this_process())


def test_caller_is_not_its_own_capacity_victim():
    out = run(SelfSpared())
    assert value(out.other_signal) == PREEMPTED
    assert value(out.other_at) == 1.0
    assert value(out.self_held) == 1
    assert value(out.in_use_after) == 0


class Holding(cb.Model):
    shop: cb.Ref["Released"]

    def __init__(self, shop):
        self.shop = shop

    @cb.process
    def hold_forever(self):
        self.shop.machine.acquire()
        cb.hold(100.0)


class Waiting(cb.Model):
    shop: cb.Ref["Released"]

    def __init__(self, shop):
        self.shop = shop

    @cb.process
    def wait_forever(self):
        cb.hold(1.0)
        self.shop.machine.acquire()
        self.shop.waiter_got = 1


class Released(cb.Model):
    """Releasing a model stops its process: a waiting one leaves the queue
    (t=3), a holding one gives its unit back (t=5) to Last, queued at t=2."""
    holder: Holding
    waiter: Waiting
    machine: cb.Resource = cb.Resource(capacity=1)
    waiter_got: cb.Output[int]
    last_at: cb.Output[float]
    held_after: cb.Output[int]

    def __init__(self):
        self.holder = Holding(self)
        self.waiter = Waiting(self)

    @cb.process
    def last(self):
        cb.hold(2.0)
        self.machine.acquire()
        self.last_at = cb.now()
        self.held_after = self.machine.held(cb.this_process())
        cb.hold(1.0)
        self.machine.release()

    @cb.process
    def releaser(self):
        cb.hold(3.0)
        cb.release(self.waiter)
        cb.hold(2.0)
        cb.release(self.holder)


def test_stopped_waiters_and_holders_leave_the_resource():
    out = run(Released(), replications=3)
    assert out.waiter_got.values.tolist() == [[0.0, 0.0, 0.0]]
    assert out.last_at.values.tolist() == [[5.0, 5.0, 5.0]]
    assert out.held_after.values.tolist() == [[1.0, 1.0, 1.0]]


class Crowd(cb.Model):
    """Processes waiting, holding and partially served when the trial is
    abandoned by releasing more than is held."""
    machine: cb.Resource = cb.Resource(capacity=3)
    bad: cb.Param[int] = 1

    @cb.process
    def holder(self):
        self.machine.acquire(2)
        cb.hold(100.0)

    @cb.process
    def partial(self):
        cb.hold(1.0)
        self.machine.acquire(3, priority=4)

    @cb.process
    def waiter(self):
        cb.hold(2.0)
        self.machine.acquire()

    @cb.process
    def culprit(self):
        cb.hold(3.0)
        if self.bad:
            self.machine.release(1)


def test_abandoned_trials_with_waiters_leak_nothing():
    model = Crowd()
    results = cb.Experiment(model, replications=200).run(workers=1)
    assert results.failed.all()
    assert "release(1) exceeds the units held" in results.failure_reasons[0][0]
    model.bad = 0
    assert not cb.Experiment(model, replications=50,
                             window=cb.Window(duration=10)).run(workers=1).failed.any()


class BadAmount(cb.Model):
    machine: cb.Resource = cb.Resource(capacity=3)
    amount: cb.Param[int] = 0

    @cb.process
    def ask(self):
        self.machine.acquire(self.amount)


def test_nonpositive_amounts_fail_the_trial():
    for amount in (0, -1):
        model = BadAmount()
        model.amount = amount
        results = cb.Experiment(model).run(workers=1)
        assert results.failed.all()
        assert "needs a positive amount" in results.failure_reasons[0][0]
