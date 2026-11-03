"""cb.acquire_any: one unit from whichever of several resources grants first,
with a fair place in every member's queue. Exact timestamps throughout."""

import pytest

import cimba as cb

PREEMPTED = -1
INTERRUPTED = -2
TIMEOUT = -5


def run(model, **kwargs):
    result = cb.Experiment(model, **kwargs).run(workers=1)
    assert not result.failed.any(), result.failure_reasons
    return result[model]


def value(output):
    return output.values.tolist()[0][0]


class Desk(cb.Model):
    """Clerks and seniors, one each. Holders keep the clerk until t=10 and
    the senior until t=12. Any (acquire_any over both) and Direct (clerks
    only) arrive in the order the parameters say; Later queues for the senior
    at t=3. Whoever gets a unit keeps it for 5."""
    clerks: cb.Resource = cb.Resource(capacity=1)
    seniors: cb.Resource = cb.Resource(capacity=1)
    any_arrives: cb.Param[float] = 1.0
    direct_arrives: cb.Param[float] = 2.0
    any_priority: cb.Param[int] = -1
    any_got: cb.Output[int]
    any_at: cb.Output[float]
    direct_at: cb.Output[float]
    later_at: cb.Output[float]

    @cb.process(priority=9)
    def holders(self):
        self.clerks.acquire()
        self.seniors.acquire()
        cb.hold(10.0)
        self.clerks.release()
        cb.hold(2.0)
        self.seniors.release()

    @cb.process
    def any(self):
        cb.hold(self.any_arrives)
        if self.any_priority < 0:
            self.any_got = cb.acquire_any((self.clerks, self.seniors))
        else:
            self.any_got = cb.acquire_any((self.clerks, self.seniors),
                                          priority=self.any_priority)
        self.any_at = cb.now()
        cb.hold(5.0)
        if self.any_got == 0:
            self.clerks.release()
        else:
            self.seniors.release()

    @cb.process
    def direct(self):
        cb.hold(self.direct_arrives)
        self.clerks.acquire()
        self.direct_at = cb.now()
        cb.hold(5.0)
        self.clerks.release()

    @cb.process
    def later(self):
        cb.hold(3.0)
        self.seniors.acquire()
        self.later_at = cb.now()
        self.seniors.release()


def test_earlier_acquire_any_is_served_first_and_withdrawn_elsewhere():
    out = run(Desk())
    assert value(out.any_got) == 0
    assert value(out.any_at) == 10.0
    assert value(out.direct_at) == 15.0
    # Any left the seniors' queue when the clerk granted it
    assert value(out.later_at) == 12.0


def test_earlier_direct_acquirer_is_served_first():
    model = Desk()
    model.any_arrives, model.direct_arrives = 2.0, 1.0
    out = run(model)
    assert value(out.direct_at) == 10.0
    # Still ahead of Later in the seniors' queue
    assert value(out.any_got) == 1
    assert value(out.any_at) == 12.0
    assert value(out.later_at) == 17.0


def test_explicit_priority_goes_ahead_of_earlier_arrivals():
    model = Desk()
    model.any_arrives, model.direct_arrives = 2.0, 1.0
    model.any_priority = 3
    out = run(model)
    assert value(out.any_got) == 0
    assert value(out.any_at) == 10.0
    assert value(out.direct_at) == 15.0
    assert value(out.later_at) == 12.0


class Raised(cb.Model):
    """Direct (t=0) queues for the clerk; Any (t=1, process priority 0)
    queues behind it in both queues. At t=2 Any's process priority goes to
    4, which re-ranks it in both queues, so it takes the clerk at t=10."""
    clerks: cb.Resource = cb.Resource(capacity=1)
    seniors: cb.Resource = cb.Resource(capacity=0)
    any_pointer: int = 0
    any_at: cb.Output[float]
    direct_at: cb.Output[float]

    @cb.process(priority=9)
    def holder(self):
        self.clerks.acquire()
        cb.hold(10.0)
        self.clerks.release()

    @cb.process
    def direct(self):
        self.clerks.acquire()
        self.direct_at = cb.now()
        self.clerks.release()

    @cb.process
    def any(self):
        self.any_pointer = cb.this_process().pointer
        cb.hold(1.0)
        cb.acquire_any((self.clerks, self.seniors))
        self.any_at = cb.now()
        cb.hold(1.0)
        self.clerks.release()

    @cb.process
    def raiser(self):
        cb.hold(2.0)
        cb.Process(self.any_pointer).priority_set(4)


def test_tracking_priority_follows_priority_set_in_every_queue():
    out = run(Raised())
    assert value(out.any_at) == 10.0
    assert value(out.direct_at) == 11.0


class Free(cb.Model):
    a: cb.Resource = cb.Resource(capacity=1)
    b: cb.Resource = cb.Resource(capacity=1)
    first: cb.Output[int]
    second: cb.Output[int]
    third: cb.Output[int]

    @cb.process
    def take(self):
        self.first = cb.acquire_any((self.b, self.a))   # both free: b, listed first
        self.second = cb.acquire_any((self.b, self.a))  # only a is free
        self.b.release()
        self.third = cb.acquire_any((self.a, self.b, self.a))


def test_first_listed_free_resource_wins():
    out = run(Free())
    assert value(out.first) == 0
    assert value(out.second) == 1
    assert value(out.third) == 1


class StaleAny(cb.Model):
    """Holder releases the clerk at t=10, granting Any (waiting since t=1 in
    both queues). Before Any's wakeup runs, Cat interrupts it. Any returns the
    signal holding nothing; Direct, behind it in the clerks' queue, gets the
    clerk at t=10 and Later, behind it in the seniors', the senior at t=12."""
    clerks: cb.Resource = cb.Resource(capacity=1)
    seniors: cb.Resource = cb.Resource(capacity=1)
    any_pointer: int = 0
    when: cb.Param[float] = 10.0
    any_signal: cb.Output[int]
    any_held: cb.Output[int]
    any_back: cb.Output[float]
    direct_at: cb.Output[float]
    later_at: cb.Output[float]

    @cb.process(priority=5)
    def holder(self):
        self.clerks.acquire()
        self.seniors.acquire()
        cb.hold(10.0)
        self.clerks.release()
        cb.hold(2.0)
        self.seniors.release()

    @cb.process
    def any(self):
        self.any_pointer = cb.this_process().pointer
        cb.hold(1.0)
        self.any_signal = cb.acquire_any((self.clerks, self.seniors))
        me = cb.this_process()
        self.any_held = self.clerks.held(me) + self.seniors.held(me)
        self.any_back = cb.now()
        cb.hold(100.0)

    @cb.process
    def direct(self):
        cb.hold(2.0)
        self.clerks.acquire()
        self.direct_at = cb.now()
        cb.hold(1.0)
        self.clerks.release()

    @cb.process
    def later(self):
        cb.hold(3.0)
        self.seniors.acquire()
        self.later_at = cb.now()
        self.seniors.release()

    @cb.process(priority=3)
    def cat(self):
        cb.hold(self.when)
        cb.Process(self.any_pointer).interrupt(INTERRUPTED, 4)


def test_interrupt_after_grant_leaves_no_stale_entries():
    out = run(StaleAny())
    assert value(out.any_signal) == INTERRUPTED
    assert value(out.any_held) == 0
    assert value(out.any_back) == 10.0
    assert value(out.direct_at) == 10.0
    assert value(out.later_at) == 12.0


def test_interrupt_while_waiting_leaves_no_stale_entries():
    model = StaleAny()
    model.when = 4.0
    out = run(model)
    assert value(out.any_signal) == INTERRUPTED
    assert value(out.any_held) == 0
    assert value(out.any_back) == 4.0
    assert value(out.direct_at) == 10.0
    assert value(out.later_at) == 12.0


class Bumped(cb.Model):
    """Any gets the clerk through acquire_any at t=0; Boss (priority 5)
    preempts it at t=2: the unit is an ordinary holding of the clerks."""
    clerks: cb.Resource = cb.Resource(capacity=1)
    seniors: cb.Resource = cb.Resource(capacity=0)
    got: cb.Output[int]
    hold_signal: cb.Output[int]
    held_after: cb.Output[int]
    boss_at: cb.Output[float]

    @cb.process
    def any(self):
        self.got = cb.acquire_any((self.clerks, self.seniors))
        self.hold_signal = cb.hold(10.0)
        self.held_after = self.clerks.held(cb.this_process())

    @cb.process(priority=5)
    def boss(self):
        cb.hold(2.0)
        self.clerks.preempt()
        self.boss_at = cb.now()
        self.clerks.release()


def test_an_acquire_any_holding_can_be_preempted():
    out = run(Bumped())
    assert value(out.got) == 0
    assert value(out.hold_signal) == PREEMPTED
    assert value(out.held_after) == 0
    assert value(out.boss_at) == 2.0


class Staffed(cb.Model):
    """Neither role is staffed at first. Any waits from t=1; the seniors
    open with one unit at t=4, which grants it."""
    clerks: cb.Resource = cb.Resource(capacity=0)
    seniors: cb.Resource = cb.Resource(capacity=0)
    got: cb.Output[int]
    at: cb.Output[float]

    @cb.process
    def any(self):
        cb.hold(1.0)
        self.got = cb.acquire_any((self.clerks, self.seniors))
        self.at = cb.now()
        self.seniors.release()

    @cb.process
    def staffing(self):
        cb.hold(4.0)
        self.seniors.set_capacity(1)


def test_raising_capacity_grants_a_waiting_acquire_any():
    out = run(Staffed())
    assert value(out.got) == 1
    assert value(out.at) == 4.0


class TimedOut(cb.Model):
    """A process timer ends the wait at t=4 with its signal; the clerk then
    goes to Direct, behind Any, at t=10."""
    clerks: cb.Resource = cb.Resource(capacity=1)
    seniors: cb.Resource = cb.Resource(capacity=0)
    signal: cb.Output[int]
    at: cb.Output[float]
    direct_at: cb.Output[float]

    @cb.process(priority=9)
    def holder(self):
        self.clerks.acquire()
        cb.hold(10.0)
        self.clerks.release()

    @cb.process
    def any(self):
        cb.hold(1.0)
        cb.this_process().timer_set(3.0, TIMEOUT)
        self.signal = cb.acquire_any((self.clerks, self.seniors))
        self.at = cb.now()

    @cb.process
    def direct(self):
        cb.hold(2.0)
        self.clerks.acquire()
        self.direct_at = cb.now()


def test_a_process_timer_bounds_the_wait():
    out = run(TimedOut())
    assert value(out.signal) == TIMEOUT
    assert value(out.at) == 4.0
    assert value(out.direct_at) == 10.0


class Crowd(cb.Model):
    """acquire_any waiters in several queues when the trial is abandoned by
    releasing more than is held."""
    a: cb.Resource = cb.Resource(capacity=1)
    b: cb.Resource = cb.Resource(capacity=1)
    bad: cb.Param[int] = 1

    @cb.process(copies=3)
    def holder(self):
        cb.acquire_any((self.a, self.b))
        cb.hold(100.0)

    @cb.process
    def ranked(self):
        cb.hold(1.0)
        cb.acquire_any((self.b, self.a), priority=4)

    @cb.process
    def culprit(self):
        cb.hold(3.0)
        if self.bad:
            self.a.release(1)


def test_abandoned_trials_with_acquire_any_waiters_leak_nothing():
    model = Crowd()
    results = cb.Experiment(model, replications=200).run(workers=1)
    assert results.failed.all()
    assert "release(1) exceeds the units held" in results.failure_reasons[0][0]
    model.bad = 0
    assert not cb.Experiment(model, replications=50,
                             window=cb.Window(duration=10)).run(workers=1).failed.any()


class NotATuple(cb.Model):
    a: cb.Resource = cb.Resource(capacity=1)

    @cb.process
    def take(self):
        cb.acquire_any(self.a)


def test_acquire_any_needs_a_tuple_of_resources():
    with pytest.raises(Exception, match="tuple of Resources"):
        cb.Experiment(NotATuple()).run(workers=1)


class Outside(cb.Model):
    a: cb.Resource = cb.Resource(capacity=0)

    @cb.on_start
    def start(self):
        cb.acquire_any((self.a,))


def test_acquire_any_outside_a_process_fails_the_trial():
    results = cb.Experiment(Outside()).run(workers=1)
    assert results.failed.all()
    assert "acquire_any() cannot hold a resource" in results.failure_reasons[0][0]


def test_process_graph_draws_acquire_any_from_every_member():
    from cimba.diagrams import process_graph

    edges = {(e.source, e.label, e.target) for e in process_graph(Desk()).edges}
    assert ("desk.clerks", "acquire any", "desk.any") in edges
    assert ("desk.seniors", "acquire any", "desk.any") in edges
