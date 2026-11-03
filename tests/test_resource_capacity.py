"""Time-varying Resource capacity: set_capacity, capacity, mean_capacity, utilization."""

import math

import pytest

import cimba as cb
from cimba.compiler import ModelCompileError

PREEMPTED = -1


def run(model, **kwargs):
    result = cb.Experiment(model, **kwargs).run(workers=1)
    assert not result.failed.any(), result.failure_reasons
    return result[model]


def value(output):
    return output.values.tolist()[0][0]


class Raise(cb.Model):
    """Holder keeps the only unit. W1, W2, W3 queue at t=1, 2, 3; at t=5 the
    capacity goes to 3, so W1 and W2 start at t=5 and W3 when W1 is done."""
    clerks: cb.Resource = cb.Resource(capacity=1)
    seq: int = 0
    w1_at: cb.Output[float]
    w2_at: cb.Output[float]
    w3_at: cb.Output[float]
    w1_seq: cb.Output[int]
    w2_seq: cb.Output[int]
    capacity_after: cb.Output[int]

    @cb.process
    def holder(self):
        self.clerks.acquire()
        cb.hold(100.0)
        self.clerks.release()

    @cb.process
    def w1(self):
        cb.hold(1.0)
        self.clerks.acquire()
        self.seq += 1
        self.w1_seq = self.seq
        self.w1_at = cb.now()
        cb.hold(10.0)
        self.clerks.release()

    @cb.process
    def w2(self):
        cb.hold(2.0)
        self.clerks.acquire()
        self.seq += 1
        self.w2_seq = self.seq
        self.w2_at = cb.now()
        cb.hold(20.0)
        self.clerks.release()

    @cb.process
    def w3(self):
        cb.hold(3.0)
        self.clerks.acquire()
        self.w3_at = cb.now()
        self.clerks.release()

    @cb.process
    def calendar(self):
        cb.hold(5.0)
        self.clerks.set_capacity(3)
        self.capacity_after = self.clerks.capacity()


def test_raising_capacity_grants_waiters_in_queue_order():
    out = run(Raise())
    assert value(out.w1_at) == 5.0
    assert value(out.w2_at) == 5.0
    assert (value(out.w1_seq), value(out.w2_seq)) == (1, 2)
    assert value(out.w3_at) == 15.0
    assert value(out.capacity_after) == 3


class Finish(cb.Model):
    """A, B, C hold one unit each from t=0 for 10, 20, 30. At t=1 the capacity
    drops to 1 with "finish": they keep their units and W, queued at t=2, waits
    until in_use drops below 1 at t=30."""
    clerks: cb.Resource = cb.Resource(capacity=3)
    a_signal: cb.Output[int]
    w_at: cb.Output[float]
    available_at_2: cb.Output[int]
    available_at_25: cb.Output[int]
    available_at_31: cb.Output[int]

    @cb.process
    def a(self):
        self.clerks.acquire()
        self.a_signal = cb.hold(10.0)
        self.clerks.release()

    @cb.process
    def b(self):
        self.clerks.acquire()
        cb.hold(20.0)
        self.clerks.release()

    @cb.process
    def c(self):
        self.clerks.acquire()
        cb.hold(30.0)
        self.clerks.release()

    @cb.process
    def w(self):
        cb.hold(2.0)
        self.clerks.acquire()
        self.w_at = cb.now()
        cb.hold(5.0)
        self.clerks.release()

    @cb.process
    def calendar(self):
        cb.hold(1.0)
        self.clerks.set_capacity(1, mode="finish")

    @cb.process
    def probe(self):
        cb.hold(2.5)
        self.available_at_2 = self.clerks.available()
        cb.hold(22.5)
        self.available_at_25 = self.clerks.available()
        cb.hold(6.0)
        self.available_at_31 = self.clerks.available()


def test_lowering_with_finish_lets_holders_keep_their_units():
    out = run(Finish())
    assert value(out.a_signal) == 0
    assert value(out.available_at_2) == 0
    assert value(out.available_at_25) == 0
    assert value(out.w_at) == 30.0
    # W holds the single unit at t=31
    assert value(out.available_at_31) == 0


class Preempt(cb.Model):
    """L1 (priority 0, t=0) and L2 (priority 0, t=1) hold at their process
    priority; R (priority 0) holds at request priority 9 from t=2; H (priority
    5) holds from t=3. At t=5 the capacity drops to 1 with "preempt": L2, then
    L1, then H are thrown out, R keeps its unit. L1 re-acquires and gets back
    in when R finishes at t=12."""
    clerks: cb.Resource = cb.Resource(capacity=4)
    seq: int = 0
    l1_signal: cb.Output[int]
    l1_seq: cb.Output[int]
    l2_signal: cb.Output[int]
    l2_seq: cb.Output[int]
    h_signal: cb.Output[int]
    h_seq: cb.Output[int]
    r_signal: cb.Output[int]
    l1_held_after: cb.Output[int]
    l1_back_at: cb.Output[float]
    l1_remaining: cb.Output[float]

    @cb.process
    def l1(self):
        remaining = 20.0
        self.clerks.acquire()
        start = cb.now()
        self.l1_signal = cb.hold(remaining)
        self.seq += 1
        self.l1_seq = self.seq
        self.l1_held_after = self.clerks.held(cb.this_process())
        if self.l1_signal == PREEMPTED:
            remaining -= cb.now() - start
            self.l1_remaining = remaining
            self.clerks.acquire()
            self.l1_back_at = cb.now()
            cb.hold(remaining)
        self.clerks.release()

    @cb.process
    def l2(self):
        cb.hold(1.0)
        self.clerks.acquire()
        self.l2_signal = cb.hold(20.0)
        self.seq += 1
        self.l2_seq = self.seq

    @cb.process
    def r(self):
        cb.hold(2.0)
        self.clerks.acquire(priority=9)
        self.r_signal = cb.hold(10.0)
        self.clerks.release()

    @cb.process(priority=5)
    def h(self):
        cb.hold(3.0)
        self.clerks.acquire()
        self.h_signal = cb.hold(20.0)
        self.seq += 1
        self.h_seq = self.seq

    @cb.process
    def calendar(self):
        cb.hold(5.0)
        self.clerks.set_capacity(1, mode="preempt")


def test_lowering_with_preempt_takes_lowest_rank_most_recent_first():
    out = run(Preempt())
    assert value(out.r_signal) == 0
    assert value(out.l1_signal) == PREEMPTED
    assert value(out.l2_signal) == PREEMPTED
    assert value(out.h_signal) == PREEMPTED
    # Victims wake at their own priority: H (5) first, then L2, L1 (0) in
    # the order they were preempted.
    assert (value(out.h_seq), value(out.l2_seq), value(out.l1_seq)) == (1, 2, 3)
    assert value(out.l1_held_after) == 0
    assert value(out.l1_remaining) == 15.0
    assert value(out.l1_back_at) == 12.0


class PreemptOrder(cb.Model):
    """Victim order alone: three equal holders at t=0, 1, 2, capacity 3 -> 1
    with "preempt" at t=5. The two most recent go, the first stays."""
    clerks: cb.Resource = cb.Resource(capacity=3)
    first: cb.Output[int]
    second: cb.Output[int]
    third: cb.Output[int]

    @cb.process
    def p1(self):
        self.clerks.acquire()
        self.first = cb.hold(10.0)

    @cb.process
    def p2(self):
        cb.hold(1.0)
        self.clerks.acquire()
        self.second = cb.hold(10.0)

    @cb.process
    def p3(self):
        cb.hold(2.0)
        self.clerks.acquire()
        self.third = cb.hold(10.0)

    @cb.process
    def calendar(self):
        cb.hold(5.0)
        self.clerks.set_capacity(1, mode="preempt")


def test_preempt_keeps_the_oldest_equal_holder():
    out = run(PreemptOrder())
    assert value(out.first) == 0
    assert value(out.second) == PREEMPTED
    assert value(out.third) == PREEMPTED


class PreemptRoom(cb.Model):
    """Big holds 3 of 4 units (most recent), Small 1. Lowering to 2 with
    "preempt" throws out Big alone, leaving room for Waiter's 1 unit at t=5."""
    clerks: cb.Resource = cb.Resource(capacity=4)
    big_signal: cb.Output[int]
    small_signal: cb.Output[int]
    waiter_at: cb.Output[float]

    @cb.process
    def small(self):
        self.clerks.acquire(1)
        self.small_signal = cb.hold(100.0)

    @cb.process
    def big(self):
        cb.hold(1.0)
        self.clerks.acquire(3)
        self.big_signal = cb.hold(100.0)

    @cb.process
    def waiter(self):
        cb.hold(2.0)
        self.clerks.acquire(1)
        self.waiter_at = cb.now()

    @cb.process
    def calendar(self):
        cb.hold(5.0)
        self.clerks.set_capacity(2, mode="preempt")


def test_preempted_room_below_the_new_capacity_goes_to_waiters():
    out = run(PreemptRoom())
    assert value(out.big_signal) == PREEMPTED
    assert value(out.small_signal) == 0
    assert value(out.waiter_at) == 5.0


class Closed(cb.Model):
    """The role starts with no staff, opens with 2 at t=10 and closes again
    at t=20 with "finish". W1 wants 2 units, W2 one."""
    clerks: cb.Resource = cb.Resource(capacity=0)
    w1_at: cb.Output[float]
    w2_at: cb.Output[float]

    @cb.process
    def w1(self):
        cb.hold(1.0)
        self.clerks.acquire(2)
        self.w1_at = cb.now()
        cb.hold(5.0)
        self.clerks.release(2)

    @cb.process
    def w2(self):
        cb.hold(2.0)
        self.clerks.acquire()
        self.w2_at = cb.now()
        self.clerks.release()

    @cb.process
    def calendar(self):
        cb.hold(10.0)
        self.clerks.set_capacity(2)
        cb.hold(10.0)
        self.clerks.set_capacity(0)


def test_capacity_zero_and_back():
    out = run(Closed())
    assert value(out.w1_at) == 10.0
    assert value(out.w2_at) == 15.0


class Stats(cb.Model):
    """Capacity 2 until t=10, 4 until t=30, then 1. One unit is in use over
    [5, 25). Window: warmup 5, duration 20, so it covers [5, 25):
    mean capacity (5*2 + 15*4) / 20 = 3.5, mean in use 1, utilization 1/3.5."""
    clerks: cb.Resource = cb.Resource(capacity=2)
    mean_capacity: cb.Output[float]
    mean_in_use: cb.Output[float]
    utilization: cb.Output[float]

    @cb.process
    def user(self):
        cb.hold(5.0)
        self.clerks.acquire()
        cb.hold(20.0)
        self.clerks.release()

    @cb.process
    def calendar(self):
        cb.hold(10.0)
        self.clerks.set_capacity(4)
        cb.hold(20.0)
        self.clerks.set_capacity(1)

    @cb.on_end
    def collect(self):
        self.mean_capacity = self.clerks.mean_capacity()
        self.mean_in_use = self.clerks.mean_in_use()
        self.utilization = self.clerks.utilization()


def test_mean_capacity_and_utilization_over_the_window():
    out = run(Stats(), window=cb.Window(warmup=5.0, duration=20.0))
    assert value(out.mean_capacity) == pytest.approx(3.5, abs=1e-12)
    assert value(out.mean_in_use) == pytest.approx(1.0, abs=1e-12)
    assert value(out.utilization) == pytest.approx(1.0 / 3.5, abs=1e-12)


class NoCapacity(cb.Model):
    clerks: cb.Resource = cb.Resource(capacity=0)
    mean_capacity: cb.Output[float]
    utilization: cb.Output[float]

    @cb.on_end
    def collect(self):
        self.mean_capacity = self.clerks.mean_capacity()
        self.utilization = self.clerks.utilization()


def test_utilization_is_nan_without_capacity():
    out = run(NoCapacity(), window=cb.Window(duration=10.0))
    assert value(out.mean_capacity) == 0.0
    assert math.isnan(value(out.utilization))


class Same(cb.Model):
    """Setting the current capacity with "preempt" changes nothing."""
    clerks: cb.Resource = cb.Resource(capacity=1)
    signal: cb.Output[int]

    @cb.process
    def holder(self):
        self.clerks.acquire()
        self.signal = cb.hold(10.0)
        self.clerks.release()

    @cb.process
    def calendar(self):
        cb.hold(5.0)
        self.clerks.set_capacity(1, mode="preempt")


def test_setting_the_same_capacity_is_a_no_op():
    out = run(Same())
    assert value(out.signal) == 0


class Customer(cb.Model):
    office: cb.Ref["Shifts"]

    @cb.process
    def visit(self):
        self.office.clerks.acquire()
        cb.hold(2.0)
        self.office.clerks.release()
        self.office.served += 1


class Shifts(cb.Model):
    """The calendar from the policies guide: 3 clerks for 8 hours, 1 for 4,
    closed for 12. Customers arrive every hour and need 2 hours; each shift
    change lets the clerks finish the customer at hand."""
    clerks: cb.Resource = cb.Resource(capacity=0)
    day: cb.Param[int] = 3
    evening: cb.Param[int] = 1
    served: int = 0
    served_out: cb.Output[int]
    mean_capacity: cb.Output[float]
    utilization: cb.Output[float]

    @cb.process(priority=10)
    def shifts(self):
        while True:
            self.clerks.set_capacity(self.day)
            cb.hold(8.0)
            self.clerks.set_capacity(self.evening)
            cb.hold(4.0)
            self.clerks.set_capacity(0)
            cb.hold(12.0)

    @cb.process
    def arrivals(self):
        while True:
            cb.spawn(Customer, office=self)
            cb.hold(1.0)

    @cb.on_end
    def collect(self):
        self.served_out = self.served
        self.mean_capacity = self.clerks.mean_capacity()
        self.utilization = self.clerks.utilization()


def test_shift_calendar_process():
    out = run(Shifts(), window=cb.Window(duration=48.0))
    assert value(out.mean_capacity) == pytest.approx((3 * 8 + 1 * 4) / 24.0)
    # Day 1: 1 unit in use over [0, 1), 2 over [1, 8), 1 over [8, 13): 20
    # unit-hours. Day 2 opens with a backlog: 3 busy over [24, 32), 1 over
    # [32, 36): 28. Capacity offered: 28 unit-hours a day.
    assert value(out.utilization) == pytest.approx(48.0 / 56.0)
    assert value(out.served_out) == 10 + 14


def test_mode_must_be_a_known_literal():
    class Unknown(cb.Model):
        clerks: cb.Resource

        @cb.process
        def act(self):
            self.clerks.set_capacity(0, mode="drain")  # pyright: ignore[reportArgumentType]

    with pytest.raises(ModelCompileError, match="'finish' or 'preempt'"):
        cb.Experiment(Unknown()).run(workers=1)


def test_mode_must_be_a_literal():
    class Variable(cb.Model):
        clerks: cb.Resource

        @cb.process
        def act(self):
            mode = "finish"
            if cb.now() > 1.0:
                mode = "preempt"
            self.clerks.set_capacity(0, mode=mode)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ModelCompileError, match="string literal"):
        cb.Experiment(Variable()).run(workers=1)


class Negative(cb.Model):
    """A negative capacity fails the trial, also while units are held."""
    clerks: cb.Resource = cb.Resource(capacity=2)
    target: cb.Param[int] = -1

    @cb.process
    def holder(self):
        self.clerks.acquire()
        cb.hold(10.0)
        self.clerks.release()

    @cb.process
    def waiter(self):
        cb.hold(0.5)
        self.clerks.acquire(2)

    @cb.process
    def calendar(self):
        cb.hold(1.0)
        self.clerks.set_capacity(self.target)


def test_negative_capacity_fails_the_trial():
    model = Negative()
    results = cb.Experiment(model, replications=2).run(workers=1)
    assert results.failed.all()
    assert "set_capacity(-1) needs a nonnegative capacity" in results.failure_reasons[0][0]
    # The same model still runs afterwards: the abandoned trials leaked nothing.
    model.target = 3
    assert not cb.Experiment(model).run(workers=1).failed.any()


def test_initial_capacity_is_configuration():
    resource = cb.Resource(capacity=0)
    assert resource.initial_capacity == 0
    resource.initial_capacity = 4
    with pytest.raises(ValueError, match="nonnegative"):
        resource.initial_capacity = -1
    with pytest.raises(ValueError, match="nonnegative"):
        cb.Resource(capacity=-1)
    with pytest.raises(AttributeError, match="initial_capacity"):
        resource.capacity = 2  # pyright: ignore[reportAttributeAccessIssue]


class HookSet(cb.Model):
    """A hook can change the capacity too: it never waits."""
    clerks: cb.Resource = cb.Resource(capacity=1)
    seen: cb.Output[int]

    @cb.on_start
    def staff(self):
        self.clerks.set_capacity(4, mode="preempt")

    @cb.process
    def look(self):
        self.seen = self.clerks.capacity()


def test_hooks_can_set_the_capacity():
    out = run(HookSet())
    assert value(out.seen) == 4
