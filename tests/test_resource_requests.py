"""Per-request priority on Resource.acquire/preempt, independent of the process."""

import pytest

import cimba as cb
from cimba.compiler import ModelCompileError

PREEMPTED = -1


def run(model):
    result = cb.Experiment(model).run(workers=1)
    assert not result.failed.any()
    return result[model]


class QueueOrder(cb.Model):
    """Holder (priority 3) releases at t=10; Mark (priority 1) wakes at t=10.

    Ranked (process priority 0) waits behind Plain (process priority 5) unless
    its request priority puts it in front.
    """
    machine: cb.Resource = cb.Resource(capacity=1)
    request: cb.Param[int] = -1
    seq: int = 0
    plain_at: cb.Output[float]
    ranked_at: cb.Output[float]
    ranked_seq: cb.Output[int]
    mark_seq: cb.Output[int]

    @cb.process(priority=3)
    def holder(self):
        self.machine.acquire()
        cb.hold(10.0)
        self.machine.release()

    @cb.process(priority=5)
    def plain(self):
        cb.hold(1.0)
        self.machine.acquire()
        self.plain_at = cb.now()
        cb.hold(5.0)
        self.machine.release()

    @cb.process(priority=0)
    def ranked(self):
        cb.hold(2.0)
        if self.request < 0:
            self.machine.acquire(1, priority=None)
        else:
            self.machine.acquire(1, priority=self.request)
        self.seq += 1
        self.ranked_seq = self.seq
        self.ranked_at = cb.now()
        cb.hold(5.0)
        self.machine.release()

    @cb.process(priority=1)
    def mark(self):
        cb.hold(10.0)
        self.seq += 1
        self.mark_seq = self.seq


def test_request_priority_orders_queue_but_not_same_time_wakeups():
    model = QueueOrder()
    model.request = 10
    out = run(model)
    # Ranked jumps the queue...
    assert out.ranked_at.values.tolist() == [[10.0]]
    assert out.plain_at.values.tolist() == [[15.0]]
    # ...but its wakeup at t=10 still runs at process priority 0, after Mark (1).
    assert out.mark_seq.values.tolist() == [[1.0]]
    assert out.ranked_seq.values.tolist() == [[2.0]]


class PositionalPriority(QueueOrder):
    """Same as QueueOrder, with the request priority passed by position."""

    @cb.process(priority=0)
    def ranked(self):
        cb.hold(2.0)
        self.machine.acquire(1, 10)
        self.seq += 1
        self.ranked_seq = self.seq
        self.ranked_at = cb.now()
        cb.hold(5.0)
        self.machine.release()


def test_request_priority_may_be_positional():
    # priority is positional-or-keyword, like PriorityStore.put and schedule.
    out = run(PositionalPriority())
    assert out.ranked_at.values.tolist() == [[10.0]]
    assert out.plain_at.values.tolist() == [[15.0]]


def test_default_priority_follows_the_process():
    model = QueueOrder()
    model.request = -1
    out = run(model)
    assert out.plain_at.values.tolist() == [[10.0]]
    assert out.ranked_at.values.tolist() == [[15.0]]


def test_equal_request_priority_is_first_come_first_served():
    # Plain waits at its process priority 5, Ranked arrives later at 5.
    model = QueueOrder()
    model.request = 5
    out = run(model)
    assert out.plain_at.values.tolist() == [[10.0]]
    assert out.ranked_at.values.tolist() == [[15.0]]


class Reprioritized(cb.Model):
    """Fixed queues at request priority 0, Track (priority 1) and Late (0)
    queue at their process priorities. At t=5 Fixed's process priority goes
    to 50 and Late's to 20: Late moves to the front, Fixed stays put."""
    machine: cb.Resource = cb.Resource(capacity=1)
    fixed_pointer: int = 0
    late_pointer: int = 0
    fixed_at: cb.Output[float]
    track_at: cb.Output[float]
    late_at: cb.Output[float]

    @cb.process
    def holder(self):
        self.machine.acquire()
        cb.hold(10.0)
        self.machine.release()

    @cb.process
    def fixed(self):
        self.fixed_pointer = cb.this_process().pointer
        cb.hold(1.0)
        self.machine.acquire(priority=0)
        self.fixed_at = cb.now()
        cb.hold(5.0)
        self.machine.release()

    @cb.process(priority=1)
    def track(self):
        cb.hold(2.0)
        self.machine.acquire()
        self.track_at = cb.now()
        cb.hold(5.0)
        self.machine.release()

    @cb.process
    def late(self):
        self.late_pointer = cb.this_process().pointer
        cb.hold(3.0)
        self.machine.acquire()
        self.late_at = cb.now()
        cb.hold(5.0)
        self.machine.release()

    @cb.process
    def changer(self):
        cb.hold(5.0)
        cb.Process(self.fixed_pointer).priority_set(50)
        cb.Process(self.late_pointer).priority_set(20)


def test_priority_set_does_not_move_an_explicit_request():
    out = run(Reprioritized())
    assert out.late_at.values.tolist() == [[10.0]]
    assert out.track_at.values.tolist() == [[15.0]]
    assert out.fixed_at.values.tolist() == [[20.0]]


class Preemption(cb.Model):
    """Low (priority 10) holds at request priority 0, so Mid (priority 5)
    preempts it. High (priority 0) preempts Mid at request priority 7 and holds
    at 7, even after its process priority drops to -50. Top (priority 100)
    preempts at 7 too, which is not strictly higher, so it waits."""
    machine: cb.Resource = cb.Resource(capacity=1)
    high_pointer: int = 0
    low_signal: cb.Output[int]
    mid_signal: cb.Output[int]
    high_at: cb.Output[float]
    high_signal: cb.Output[int]
    top_at: cb.Output[float]

    @cb.process(priority=10)
    def low(self):
        self.machine.acquire(priority=0)
        self.low_signal = cb.hold(100.0)

    @cb.process(priority=5)
    def mid(self):
        cb.hold(1.0)
        self.machine.preempt()
        self.mid_signal = cb.hold(100.0)

    @cb.process
    def high(self):
        self.high_pointer = cb.this_process().pointer
        cb.hold(2.0)
        self.machine.preempt(priority=7)
        self.high_at = cb.now()
        self.high_signal = cb.hold(10.0)
        if self.high_signal == 0:
            self.machine.release()

    @cb.process(priority=100)
    def top(self):
        cb.hold(3.0)
        cb.Process(self.high_pointer).priority_set(-50)
        self.machine.preempt(priority=7)
        self.top_at = cb.now()
        self.machine.release()


def test_preempt_ranks_victims_by_held_priority():
    out = run(Preemption())
    assert out.low_signal.values.tolist() == [[PREEMPTED]]
    assert out.mid_signal.values.tolist() == [[PREEMPTED]]
    assert out.high_at.values.tolist() == [[2.0]]
    assert out.high_signal.values.tolist() == [[0.0]]
    assert out.top_at.values.tolist() == [[12.0]]


class Interrupted(cb.Model):
    """Waiter queues at request priority 9 and is interrupted at t=3. Other
    queues at t=4; nothing stale may stand in front of it at t=10."""
    machine: cb.Resource = cb.Resource(capacity=1)
    waiter_pointer: int = 0
    waiter_signal: cb.Output[int]
    waiter_held: cb.Output[int]
    other_at: cb.Output[float]

    @cb.process
    def holder(self):
        self.machine.acquire()
        cb.hold(10.0)
        self.machine.release()

    @cb.process
    def waiter(self):
        self.waiter_pointer = cb.this_process().pointer
        cb.hold(1.0)
        self.waiter_signal = self.machine.acquire(priority=9)
        self.waiter_held = self.machine.held(cb.this_process())
        cb.hold(100.0)

    @cb.process
    def interrupter(self):
        cb.hold(3.0)
        cb.Process(self.waiter_pointer).interrupt(7, 0)

    @cb.process
    def other(self):
        cb.hold(4.0)
        self.machine.acquire()
        self.other_at = cb.now()
        self.machine.release()


def test_interrupted_ranked_request_leaves_no_queue_entry():
    out = run(Interrupted())
    assert out.waiter_signal.values.tolist() == [[7.0]]
    assert out.waiter_held.values.tolist() == [[0.0]]
    assert out.other_at.values.tolist() == [[10.0]]


class Rollback(cb.Model):
    """Holder keeps one of two units at its process priority 0 and asks for one
    more at request priority 9. While waiting, the whole holding ranks 9; once
    that request is interrupted at t=3, the holding ranks 0 again, so Mugger
    (priority 5) can preempt it at t=4."""
    machine: cb.Resource = cb.Resource(capacity=2)
    holder_pointer: int = 0
    first_signal: cb.Output[int]
    early_signal: cb.Output[int]
    holder_signal: cb.Output[int]
    mugger_at: cb.Output[float]

    @cb.process
    def holder(self):
        self.holder_pointer = cb.this_process().pointer
        self.machine.acquire()
        self.first_signal = self.machine.acquire(priority=9)
        self.holder_signal = cb.hold(100.0)

    @cb.process(priority=6)
    def blocker(self):
        self.machine.acquire()
        cb.hold(50.0)
        self.machine.release()

    @cb.process(priority=5)
    def mugger(self):
        cb.hold(2.0)
        # While the holding ranks 9, this preempt cannot touch it; time out.
        cb.this_process().timer_set(0.5, 11)
        self.early_signal = self.machine.preempt()
        cb.hold(1.5)
        self.machine.preempt()
        self.mugger_at = cb.now()

    @cb.process
    def interrupter(self):
        cb.hold(3.0)
        cb.Process(self.holder_pointer).interrupt(7, 0)


def test_interrupted_request_restores_holding_priority():
    out = run(Rollback())
    assert out.first_signal.values.tolist() == [[7.0]]
    assert out.early_signal.values.tolist() == [[11.0]]
    assert out.holder_signal.values.tolist() == [[PREEMPTED]]
    assert out.mugger_at.values.tolist() == [[4.0]]


def test_request_priority_must_be_an_int():
    class Fractional(cb.Model):
        machine: cb.Resource

        @cb.process
        def act(self):
            self.machine.acquire(1, priority=1.5)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ModelCompileError, match="int or None priority"):
        cb.Experiment(Fractional()).run(workers=1)
