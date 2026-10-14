"""FractionalContainer is a fixed-point cmb_buffer: exact puts and gets."""

import math

import numpy as np
import pytest

import cimba as cb
from cimba.compiler import ModelCompileError
from cimba.experiments import ExperimentConfigError


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
    # Fixed point: 0.5 + 0.75 - 1.2 is exactly 0.05, not 0.04999999999999993.
    assert outcome.after_get.values.tolist() == [[0.05]]
    assert outcome.at_end.values.tolist() == [[0.05]]
    assert outcome.peak.values[0, 0] == pytest.approx(0.5)
    # Level 0 over [0, 3), 0.05 over [3, 4].
    assert outcome.mean.values[0, 0] == pytest.approx(0.0125)
    times, values = outcome.tank.trial(0, 0)
    assert times.shape == values.shape
    assert values[0] == 0.5
    assert values[-1] == 0.05


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


class ExactGet(cb.Model):
    tank: cb.FractionalContainer = cb.FractionalContainer(initial=1.0)
    obtained: cb.Output[float]

    @cb.process
    def act(self):
        self.obtained = self.tank.get(1.0)


def test_get_returns_the_amount_requested_on_an_exact_match():
    model = ExactGet()
    results = cb.Experiment(model).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].obtained.values.tolist() == [[1.0]]


class Interrupted(cb.Model):
    tank: cb.FractionalContainer = cb.FractionalContainer(initial=0.4)
    left: cb.Output[float]
    woke_at: cb.Output[float]
    obtained: cb.Output[float]

    @cb.process
    def consumer(self):
        cb.this_process().timer_set(1.0, 7)
        self.obtained = self.tank.get(1.0)
        self.woke_at = cb.now()
        self.left = self.tank.level()


def test_interrupted_get_keeps_what_it_took():
    model = Interrupted()
    results = cb.Experiment(model).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].woke_at.values.tolist() == [[1.0]]
    assert results[model].left.values.tolist() == [[0.0]]
    # Interrupted before the rest arrived: obtained is only the 0.4 it had.
    assert results[model].obtained.values.tolist() == [[0.4]]


class Drain(cb.Model):
    """The cases where plain float arithmetic drifts."""

    tank: cb.FractionalContainer
    third: cb.Output[float]
    after_thirds: cb.Output[float]
    whole_at: cb.Output[float]
    after_drain: cb.Output[float]

    @cb.process
    def act(self):
        # In floats, 0.3 - 0.1 - 0.1 is 0.09999999999999998, short of 0.1.
        self.tank.put(0.3)
        self.tank.get(0.1)
        self.tank.get(0.1)
        self.third = self.tank.get(0.1)
        self.after_thirds = self.tank.level()
        # In floats, ten 0.1s sum to 0.9999999999999999, short of 1.0.
        for _ in range(10):
            self.tank.put(0.1)
        self.tank.get(1.0)
        self.whole_at = cb.now()
        # In floats, 1000 less ten thousand 0.1s leaves a residue.
        self.tank.put(1000.0)
        for _ in range(10_000):
            self.tank.get(0.1)
        self.after_drain = self.tank.level()


def test_puts_and_gets_are_exact():
    model = Drain()
    results = cb.Experiment(model, window=cb.Window(duration=1.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    outcome = results[model]
    assert outcome.third.values.tolist() == [[0.1]]
    assert outcome.after_thirds.values.tolist() == [[0.0]]
    assert outcome.whole_at.values.tolist() == [[0.0]]
    assert outcome.after_drain.values.tolist() == [[0.0]]


class HundredMillion(cb.Model):
    tank: cb.FractionalContainer
    level_before: cb.Output[float]
    obtained: cb.Output[float]
    got_at: cb.Output[float]

    @cb.process
    def act(self):
        self.tank.put(100_000_000.0)
        self.tank.put(0.3)
        self.tank.get(100_000_000.0)
        self.level_before = self.tank.level()
        self.obtained = self.tank.get(0.3)
        self.got_at = cb.now()


def test_hundred_million_then_small_get_is_exact():
    model = HundredMillion()
    results = cb.Experiment(model, window=cb.Window(duration=1.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    # 0.7.6 left 0.29999999701976776 here and the get(0.3) waited forever.
    outcome = results[model]
    assert outcome.level_before.values.tolist() == [[0.3]]
    assert outcome.obtained.values.tolist() == [[0.3]]
    assert outcome.got_at.values.tolist() == [[0.0]]


class Flour(cb.Model):
    """Random deliveries and withdrawals: the books must balance exactly."""

    silo: cb.FractionalContainer = cb.FractionalContainer(initial=500.0, capacity=2000.0)
    delivered: cb.State[float] = 0.0
    used: cb.State[float] = 0.0
    balance: cb.Output[float]
    level: cb.Output[float]

    @cb.process
    def deliveries(self):
        while True:
            cb.hold(cb.random.exponential(1.0))
            self.delivered += self.silo.put(cb.random.uniform(0.0, 50.0))

    @cb.process
    def mixing(self):
        while True:
            cb.hold(cb.random.exponential(1.0))
            self.used += self.silo.get(cb.random.uniform(0.0, 50.0))

    @cb.on_end
    def audit(self):
        self.balance = 500.0 + self.delivered - self.used
        self.level = self.silo.level()


def test_level_matches_deliveries_less_withdrawals():
    model = Flour()
    results = cb.Experiment(model, window=cb.Window(duration=5000.0),
                            replications=2).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    outcome = results[model]
    # Only summing the returned floats in the model rounds; the level does not.
    np.testing.assert_allclose(outcome.level.values, outcome.balance.values,
                               rtol=0, atol=1e-9)


class Rounded(cb.Model):
    tank: cb.FractionalContainer = cb.FractionalContainer(decimals=2)
    added: cb.Output[float]
    half: cb.Output[float]
    level: cb.Output[float]
    tiny: cb.Output[float]
    tiny_at: cb.Output[float]

    @cb.process
    def act(self):
        self.added = self.tank.put(0.123)
        self.half = self.tank.put(0.005)
        self.level = self.tank.level()
        cb.hold(1.0)
        self.tiny = self.tank.get(0.004)
        self.tiny_at = cb.now()


def test_amounts_are_rounded_to_decimals():
    model = Rounded()
    results = cb.Experiment(model, window=cb.Window(duration=2.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    outcome = results[model]
    assert outcome.added.values.tolist() == [[0.12]]
    # Half away from zero, not to even.
    assert outcome.half.values.tolist() == [[0.01]]
    assert outcome.level.values.tolist() == [[0.13]]
    # Rounds to zero hundredths, so there is nothing to wait for.
    assert outcome.tiny.values.tolist() == [[0.0]]
    assert outcome.tiny_at.values.tolist() == [[1.0]]


class Rounding(cb.Model):
    tank: cb.FractionalContainer = cb.FractionalContainer(decimals=2)
    float_sum: cb.Output[float]
    half_up: cb.Output[float]
    half_down: cb.Output[float]
    matches_level: cb.Output[bool]

    @cb.process
    def look(self):
        self.float_sum = self.tank.round(0.1 + 0.2)
        self.half_up = self.tank.round(0.005)
        self.half_down = self.tank.round(-0.005)
        self.tank.put(1.006)
        self.matches_level = self.tank.round(1.006) == self.tank.level()


def test_round_rounds_as_the_level_does():
    model = Rounding()
    results = cb.Experiment(model, window=cb.Window(duration=1.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    outcome = results[model]
    assert outcome.float_sum.values.tolist() == [[0.3]]
    # Half away from zero, either sign.
    assert outcome.half_up.values.tolist() == [[0.01]]
    assert outcome.half_down.values.tolist() == [[-0.01]]
    assert outcome.matches_level.values.tolist() == [[True]]
    # The host rounds alike.
    assert model.tank.round(0.1 + 0.2) == 0.3
    assert model.tank.round(-0.005) == -0.01


class Micrograms(cb.Model):
    """Small units are not swallowed by any tolerance: 2e-10 is not 1e-9."""

    tank: cb.FractionalContainer = cb.FractionalContainer(initial=2e-10, decimals=12)
    got: cb.Output[float]
    got_at: cb.Output[float]

    @cb.process
    def consumer(self):
        self.got = self.tank.get(1e-9)
        self.got_at = cb.now()

    @cb.process
    def filler(self):
        cb.hold(1.0)
        self.tank.put(8e-10)


def test_a_small_shortfall_waits():
    model = Micrograms()
    results = cb.Experiment(model, window=cb.Window(duration=2.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].got_at.values.tolist() == [[1.0]]
    assert results[model].got.values.tolist() == [[1e-9]]


class Hopper(cb.Model):
    hopper: cb.FractionalContainer = cb.FractionalContainer(initial=0.8, capacity=1.0)
    added: cb.Output[float]
    put_done: cb.Output[float]
    level: cb.Output[float]

    @cb.process
    def filler(self):
        self.added = self.hopper.put(0.5)
        self.put_done = cb.now()

    @cb.process
    def consumer(self):
        cb.hold(1.0)
        self.hopper.get(0.4)

    @cb.on_end
    def finish(self):
        self.level = self.hopper.level()


def test_put_waits_for_space_at_capacity():
    model = Hopper()
    results = cb.Experiment(model, window=cb.Window(duration=2.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    outcome = results[model]
    # 0.2 fits at once; the other 0.3 waits until the consumer makes room.
    assert outcome.put_done.values.tolist() == [[1.0]]
    assert outcome.added.values.tolist() == [[0.5]]
    assert outcome.level.values.tolist() == [[0.9]]


class InterruptedPut(cb.Model):
    hopper: cb.FractionalContainer = cb.FractionalContainer(initial=0.25, capacity=1.0)
    added: cb.Output[float]

    @cb.process
    def filler(self):
        cb.this_process().timer_set(1.0, 7)
        self.added = self.hopper.put(1.5)


def test_interrupted_put_returns_what_it_added():
    model = InterruptedPut()
    results = cb.Experiment(model).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].added.values.tolist() == [[0.75]]


class BadAmount(cb.Model):
    tank: cb.FractionalContainer
    amount: cb.Param[float] = 1.0

    @cb.process
    def act(self):
        self.tank.put(self.amount)


@pytest.mark.parametrize(("amount", "reason"), [
    (-1.0, "put(-1) is not a finite nonnegative amount"),
    (math.nan, "put(nan) is not a finite nonnegative amount"),
    (math.inf, "put(inf) is not a finite nonnegative amount"),
    (1e14, "put(100000000000000) is too large"),
])
def test_invalid_amounts_fail_the_trial(amount, reason):
    model = BadAmount()
    model.amount = amount
    results = cb.Experiment(model).run(workers=1)
    assert results.failed.all()
    assert results.failure_reasons[0][0] == f"badamount.tank: {reason}"


class Overflow(cb.Model):
    tank: cb.FractionalContainer

    @cb.process
    def act(self):
        self.tank.put(1e13)
        self.tank.put(1e13)


def test_unlimited_level_past_its_range_fails_the_trial():
    # The second put would pass 2**64 micro-units; it fails instead of waiting.
    results = cb.Experiment(Overflow()).run(workers=1)
    assert results.failed.all()
    assert results.failure_reasons[0][0] == \
        "overflow.tank: put(10000000000000) overflows the level"


class HookOverfill(cb.Model):
    silo: cb.FractionalContainer = cb.FractionalContainer(initial=90.0, capacity=100.0)

    @cb.on_start
    def stock(self):
        self.silo.put(20.0)


class HookUnderfill(cb.Model):
    silo: cb.FractionalContainer = cb.FractionalContainer(initial=10.0)

    @cb.on_start
    def take(self):
        self.silo.get(20.0)


@pytest.mark.parametrize(("model", "reason"), [
    (HookOverfill, "hookoverfill.silo: put(20) would wait for space outside a process"),
    (HookUnderfill, "hookunderfill.silo: get(20) would wait for content outside a process"),
])
def test_a_hook_that_would_wait_fails_the_trial(model, reason):
    # A hook is not a process: waiting there used to abort the interpreter.
    results = cb.Experiment(model()).run(workers=1)
    assert results.failed.all()
    assert results.failure_reasons[0][0] == reason


def test_settings_changed_after_construction_are_rechecked():
    model = Hopper()
    model.hopper.initial = 5.0
    with pytest.raises(ExperimentConfigError, match="at least initial"):
        cb.Experiment(model).run(workers=1)


def test_constructor_validates_its_arguments():
    assert cb.FractionalContainer(initial=2).initial == 2.0
    assert cb.FractionalContainer(initial=np.float32(0.5)).initial == 0.5  # pyright: ignore[reportArgumentType]
    assert cb.FractionalContainer(initial=np.int64(3)).initial == 3.0  # pyright: ignore[reportArgumentType]
    assert cb.FractionalContainer(capacity=10, decimals=0).capacity == 10.0
    assert cb.FractionalContainer(initial=0.5, capacity=0.5, decimals=0).initial == 0.5
    for bad in (-0.5, math.nan, math.inf, True, "1"):
        with pytest.raises(ValueError, match="initial"):
            cb.FractionalContainer(initial=bad)  # pyright: ignore[reportArgumentType]
    for bad in (0, -1.0, math.nan, True, 0.004, 1e14):
        with pytest.raises(ValueError, match="capacity"):
            cb.FractionalContainer(capacity=bad, decimals=2 if bad == 0.004 else 6)  # pyright: ignore[reportArgumentType]
    for bad in (-1, 13, 2.0, True):
        with pytest.raises(ValueError, match="decimals"):
            cb.FractionalContainer(decimals=bad)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="exceed capacity"):
        cb.FractionalContainer(initial=5.0, capacity=4.0)


def test_container_rejects_float_amounts():
    class Truncating(cb.Model):
        stock: cb.Container

        @cb.process
        def act(self):
            self.stock.put(0.7)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ModelCompileError, match="FractionalContainer"):
        cb.Experiment(Truncating()).run(workers=1)
