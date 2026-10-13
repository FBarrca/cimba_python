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
    # Interrupted before the rest arrived: obtained is only the 0.4 it had,
    # well short of the 1.0 requested, and the caller can see that gap.
    assert results[model].obtained.values.tolist() == [[0.4]]


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


class Drifted(cb.Model):
    tank: cb.FractionalContainer
    got_at: cb.Output[float]
    after: cb.Output[float]
    obtained: cb.Output[float]

    @cb.process
    def act(self):
        self.tank.put(0.3)
        self.tank.get(0.1)
        self.tank.get(0.1)
        # 0.3 - 0.1 - 0.1 leaves 0.09999999999999998, a hair under 0.1.
        self.obtained = self.tank.get(0.1)
        self.got_at = cb.now()
        self.after = self.tank.level()


def test_rounding_drift_does_not_block_a_get():
    model = Drifted()
    results = cb.Experiment(model, window=cb.Window(duration=1.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    # Succeeds immediately: the shortfall is rounding error, not a real wait.
    assert results[model].got_at.values.tolist() == [[0.0]]
    assert results[model].after.values[0, 0] == pytest.approx(0.0, abs=1e-9)
    # The getter can see it received a hair less than the 0.1 it asked for:
    # tolerance makes the call succeed, but does not hide the shortfall.
    assert results[model].obtained.values[0, 0] == pytest.approx(0.1, abs=1e-9)
    assert results[model].obtained.values[0, 0] < 0.1


class HugeLevelDrift(cb.Model):
    """Summing to around 1e18 loses amounts under ~128 to rounding (the ULP
    at that magnitude). A get() for the full nominal total must not wait for
    a gap that floating-point addition itself introduced."""

    tank: cb.FractionalContainer
    got_at: cb.Output[float]

    @cb.process
    def act(self):
        third = 1e18 / 3.0
        self.tank.put(third)
        self.tank.put(third)
        self.tank.put(third)
        self.tank.get(1e18)
        self.got_at = cb.now()


def test_rounding_drift_at_huge_magnitude_does_not_block():
    model = HugeLevelDrift()
    results = cb.Experiment(model, window=cb.Window(duration=1.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].got_at.values.tolist() == [[0.0]]


class MillionUnitUnderDelivery(cb.Model):
    """A level a hair short of a round million (e.g. from upstream float
    sums) still satisfies get(1_000_000) within tolerance, but the caller
    can see it received less than it asked for."""

    tank: cb.FractionalContainer = cb.FractionalContainer(initial=1_000_000.0 - 1e-4)
    obtained: cb.Output[float]

    @cb.process
    def act(self):
        self.obtained = self.tank.get(1_000_000.0)


def test_near_million_shortfall_is_visible_in_the_obtained_amount():
    model = MillionUnitUnderDelivery()
    results = cb.Experiment(model).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    obtained = results[model].obtained.values[0, 0]
    assert obtained < 1_000_000.0
    assert obtained == pytest.approx(1_000_000.0, abs=1e-3)


class HugeLevelRealShortfall(cb.Model):
    """At the same 1e18 magnitude, a shortfall far bigger than rounding error
    (1e10, vs. a tolerance of ~1e9) must still make the getter wait."""

    tank: cb.FractionalContainer = cb.FractionalContainer(initial=1e18 - 1e10)
    got_at: cb.Output[float]

    @cb.process
    def consumer(self):
        self.tank.get(1e18)
        self.got_at = cb.now()

    @cb.process
    def filler(self):
        cb.hold(2.0)
        self.tank.put(1e10)


def test_a_real_shortfall_at_huge_magnitude_still_waits():
    model = HugeLevelRealShortfall()
    results = cb.Experiment(model, window=cb.Window(duration=3.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].got_at.values.tolist() == [[2.0]]


class BarelyOutsideTolerance(cb.Model):
    """A shortfall of 1e-7 against a want of 0.1 is ~100x the tolerance floor
    (1e-9): small enough to look like it could be drift, but far too big to
    actually be drift from a single put/get pair. Must still wait."""

    tank: cb.FractionalContainer = cb.FractionalContainer(initial=0.1 - 1e-7)
    got_at: cb.Output[float]

    @cb.process
    def consumer(self):
        self.tank.get(0.1)
        self.got_at = cb.now()

    @cb.process
    def filler(self):
        cb.hold(1.0)
        self.tank.put(1e-7)


def test_shortfall_just_outside_tolerance_still_waits():
    model = BarelyOutsideTolerance()
    results = cb.Experiment(model, window=cb.Window(duration=2.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].got_at.values.tolist() == [[1.0]]


class ZeroAmountAgainstEmptyTank(cb.Model):
    """get(0.0) on an empty tank must succeed immediately: zero is always
    "close enough" to zero, and must not be treated as a wait."""

    tank: cb.FractionalContainer
    got_at: cb.Output[float]

    @cb.process
    def act(self):
        self.tank.get(0.0)
        self.got_at = cb.now()


def test_get_zero_on_an_empty_tank_does_not_wait():
    model = ZeroAmountAgainstEmptyTank()
    results = cb.Experiment(model, window=cb.Window(duration=1.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].got_at.values.tolist() == [[0.0]]


class PartialDrainThenDrift(cb.Model):
    """A getter that first drains the tank partially (taking a real partial
    amount and parking), then is topped up to a drift-only remainder, must
    be released by tolerance rather than wait for the exact bit it lost to
    rounding while parked."""

    tank: cb.FractionalContainer = cb.FractionalContainer(initial=0.2)
    got_at: cb.Output[float]
    left: cb.Output[float]

    @cb.process
    def consumer(self):
        self.tank.get(0.3)
        self.got_at = cb.now()
        self.left = self.tank.level()

    @cb.process
    def filler(self):
        cb.hold(1.0)
        # Tops up the 0.1 still owed with three 1e-18-scale nudges whose sum,
        # by itself, straddles 0.1 only within float rounding.
        self.tank.put(0.1 - 1e-16)


def test_partial_drain_released_by_tolerance_after_waiting():
    model = PartialDrainThenDrift()
    results = cb.Experiment(model, window=cb.Window(duration=2.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].got_at.values.tolist() == [[1.0]]
    assert results[model].left.values[0, 0] == 0.0


class ShortBySomethingReal(cb.Model):
    tank: cb.FractionalContainer = cb.FractionalContainer(initial=0.09)
    got_at: cb.Output[float]

    @cb.process
    def consumer(self):
        self.tank.get(0.1)
        self.got_at = cb.now()

    @cb.process
    def filler(self):
        cb.hold(2.0)
        self.tank.put(0.01)


def test_a_real_shortfall_still_waits():
    model = ShortBySomethingReal()
    results = cb.Experiment(model, window=cb.Window(duration=3.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    # 0.01 short is nowhere near tolerance: the getter must wait for the put.
    assert results[model].got_at.values.tolist() == [[2.0]]


def test_container_rejects_float_amounts():
    class Truncating(cb.Model):
        stock: cb.Container

        @cb.process
        def act(self):
            self.stock.put(0.7)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ModelCompileError, match="FractionalContainer"):
        cb.Experiment(Truncating()).run(workers=1)
