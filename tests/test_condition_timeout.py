"""Condition deadlines own and clean up their timer on every return path."""

from math import inf, nan

import pytest

import cimba as cb


class TimedWait(cb.Model):
    gate: cb.Condition
    ready: cb.State[bool] = False
    activation: cb.Param[float] = 2.0
    timeout: cb.Param[float] = 5.0
    satisfied: cb.Output[bool]
    woke_at: cb.Output[float]
    next_wake: cb.Output[float]

    @cb.predicate
    def is_ready(self):
        return self.ready

    @cb.event
    def activate(self):
        self.ready = True
        self.gate.signal()

    @cb.process
    def waiter(self):
        if self.activation >= 0:
            cb.schedule(self.activate, self.activation)
        self.satisfied = self.gate.wait_until(self.is_ready, timeout=self.timeout)
        self.woke_at = cb.now()
        cb.hold(10.0)
        self.next_wake = cb.now()


@pytest.mark.parametrize("activation, timeout, ready, expected, woke", [
    (2.0, 5.0, False, True, 2.0),
    (5.0, 5.0, False, True, 5.0),
    (-1.0, 5.0, False, False, 5.0),
    (8.0, 5.0, False, False, 5.0),
    (2.0, 0.0, False, False, 0.0),
    (-1.0, 0.0, True, True, 0.0),
    (-1.0, 5.0, True, True, 0.0),
    (2.0, inf, False, True, 2.0),
])
def test_condition_timeout_and_following_wait(activation, timeout, ready, expected, woke):
    model = TimedWait()
    model.activation, model.timeout, model.ready = activation, timeout, ready
    result = cb.Experiment(model, replications=3).run(workers=2, on_failure="raise")
    assert result[model].satisfied.values.tolist() == [[float(expected)] * 3]
    assert result[model].woke_at.values.tolist() == [[woke] * 3]
    assert result[model].next_wake.values.tolist() == [[woke + 10.0] * 3]


@pytest.mark.parametrize("timeout", [-1.0, -inf, nan])
def test_invalid_timeout_abandons_trial(timeout):
    model = TimedWait()
    model.timeout = timeout
    result = cb.Experiment(model).run(workers=1)
    assert result.failed[0, 0]


class ApplicationTimer(TimedWait):
    signal: cb.Output[int]

    @cb.process
    def waiter(self):
        process = cb.this_process()
        process.timer_set(8.0, 77)
        cb.schedule(self.activate, self.activation)
        self.satisfied = self.gate.wait_until(self.is_ready, self.timeout)
        self.woke_at = cb.now()
        self.signal = cb.suspend()
        self.next_wake = cb.now()


@pytest.mark.parametrize("activation, timeout, satisfied, woke", [
    (2.0, 5.0, True, 2.0),
    (12.0, 5.0, False, 5.0),
])
def test_wait_leaves_existing_application_timer_intact(activation, timeout, satisfied, woke):
    model = ApplicationTimer()
    model.activation, model.timeout = activation, timeout
    result = cb.Experiment(model).run(workers=1, on_failure="raise")
    assert result[model].satisfied[0, 0] == satisfied
    assert result[model].woke_at[0, 0] == woke
    assert result[model].signal[0, 0] == 77
    assert result[model].next_wake[0, 0] == 8.0


class InterruptedWait(TimedWait):
    @cb.process
    def waiter(self):
        cb.this_process().timer_set(1.0, 42)
        self.satisfied = self.gate.wait_until(self.is_ready, timeout=5.0)
        self.woke_at = cb.now()
        cb.hold(10.0)
        self.next_wake = cb.now()


def test_interruption_cancels_the_wait_timer():
    model = InterruptedWait()
    result = cb.Experiment(model).run(workers=1, on_failure="raise")
    assert result[model].satisfied[0, 0] == 0
    assert result[model].woke_at[0, 0] == 1.0
    assert result[model].next_wake[0, 0] == 11.0


class CompetingWaiters(cb.Model):
    gate: cb.Condition
    tickets: cb.State[int] = 0
    successes: cb.Output[int]
    timed_out_at: cb.Output[float]

    @cb.predicate
    def available(self):
        return self.tickets > 0

    @cb.event
    def offer(self):
        self.tickets = 1
        self.gate.signal()

    @cb.process
    def start(self):
        cb.schedule(self.offer, 1.0)

    @cb.process(copies=2)
    def wait(self):
        if self.gate.wait_until(self.available, timeout=3.0):
            self.tickets -= 1
            self.successes += 1
        else:
            self.timed_out_at = cb.now()


def test_spurious_wakeup_rechecks_predicate_with_original_deadline():
    model = CompetingWaiters()
    result = cb.Experiment(model).run(workers=1, on_failure="raise")
    assert result[model].successes[0, 0] == 1
    assert result[model].timed_out_at[0, 0] == 3.0
