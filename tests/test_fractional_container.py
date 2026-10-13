"""FractionalContainer follows cmb_buffer's get semantics with a double level."""

import math

import numpy as np
import pytest

import cimba as cb
from cimba.compiler import ModelCompileError


class Tank(cb.Model):
    tank: cb.FractionalContainer = cb.FractionalContainer(initial=0.5)
    at_start: cb.Output[float]
    got_at: cb.Output[float]
    after_get: cb.Output[float]
    at_end: cb.Output[float]
    mean: cb.Output[float]
    peak: cb.Output[float]

    @cb.on_start
    def look(self):
        self.at_start = self.tank.level()

    @cb.process
    def consumer(self):
        self.tank.get(1.2)
        self.got_at = cb.now()
        self.after_get = self.tank.level()

    @cb.process
    def filler(self):
        for _ in range(3):
            cb.hold(1.0)
            self.tank.put(0.25)

    @cb.on_end
    def finish(self):
        self.at_end = self.tank.level()
        self.mean = self.tank.mean_level()
        self.peak = self.tank.max_level()


def test_get_takes_what_is_there_and_waits_for_the_rest():
    model = Tank()
    model.tank.capture()
    results = cb.Experiment(model, window=cb.Window(duration=4.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    outcome = results[model]
    assert outcome.at_start.values.tolist() == [[0.5]]
    # 0.5 at once, 0.25 at t=1 and t=2, the last 0.2 of the 0.25 put at t=3.
    assert outcome.got_at.values.tolist() == [[3.0]]
    assert outcome.after_get.values[0, 0] == pytest.approx(0.05)
    assert outcome.at_end.values[0, 0] == pytest.approx(0.05)
    assert outcome.peak.values[0, 0] == pytest.approx(0.5)
    # Level 0 over [0, 3), 0.05 over [3, 4].
    assert outcome.mean.values[0, 0] == pytest.approx(0.0125)
    times, values = outcome.tank.trial(0, 0)
    assert times.shape == values.shape
    assert values[0] == pytest.approx(0.5)
    assert values[-1] == pytest.approx(0.05)


class Waiters(cb.Model):
    tank: cb.FractionalContainer
    first_done: cb.Output[float]
    second_done: cb.Output[float]

    @cb.process(priority=1)
    def first(self):
        self.tank.get(1.0)
        self.first_done = cb.now()

    @cb.process
    def second(self):
        self.tank.get(0.1)
        self.second_done = cb.now()

    @cb.process
    def filler(self):
        cb.hold(1.0)
        self.tank.put(0.6)
        cb.hold(1.0)
        self.tank.put(0.6)


def test_head_of_the_queue_is_served_first():
    model = Waiters()
    results = cb.Experiment(model).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    # The first getter soaks up the first put; the second waits behind it
    # and gets its share of the leftover from the second put.
    assert results[model].first_done.values.tolist() == [[2.0]]
    assert results[model].second_done.values.tolist() == [[2.0]]


class Interrupted(cb.Model):
    tank: cb.FractionalContainer = cb.FractionalContainer(initial=0.4)
    left: cb.Output[float]
    woke_at: cb.Output[float]

    @cb.process
    def consumer(self):
        cb.this_process().timer_set(1.0, 7)
        self.tank.get(1.0)
        self.woke_at = cb.now()
        self.left = self.tank.level()


def test_interrupted_get_keeps_what_it_took():
    model = Interrupted()
    results = cb.Experiment(model).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].woke_at.values.tolist() == [[1.0]]
    assert results[model].left.values.tolist() == [[0.0]]


class BadAmount(cb.Model):
    tank: cb.FractionalContainer
    amount: cb.Param[float] = 1.0

    @cb.process
    def act(self):
        self.tank.put(self.amount)


@pytest.mark.parametrize("amount", [-1.0, math.nan, math.inf])
def test_invalid_amounts_fail_the_trial(amount):
    model = BadAmount()
    model.amount = amount
    results = cb.Experiment(model).run(workers=1)
    assert results.failed.all()


def test_initial_must_be_finite_and_nonnegative():
    assert cb.FractionalContainer(initial=2).initial == 2.0
    assert cb.FractionalContainer(initial=np.float32(0.5)).initial == 0.5  # pyright: ignore[reportArgumentType]
    assert cb.FractionalContainer(initial=np.int64(3)).initial == 3.0  # pyright: ignore[reportArgumentType]
    for bad in (-0.5, math.nan, math.inf, True, "1"):
        with pytest.raises(ValueError):
            cb.FractionalContainer(initial=bad)  # pyright: ignore[reportArgumentType]


def test_container_rejects_float_amounts():
    class Truncating(cb.Model):
        stock: cb.Container

        @cb.process
        def act(self):
            self.stock.put(0.7)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ModelCompileError, match="FractionalContainer"):
        cb.Experiment(Truncating()).run(workers=1)
