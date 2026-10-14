"""FractionalContainer keeps an exact fixed-point level.

The level is an int64 count of 10**-decimals steps; amounts are rounded half
away from zero to a step once, on entry.
"""

import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pytest

import cimba as cb
from cimba.compiler import ModelCompileError

PAIRS = [(digits, decimals) for digits in range(1, 19) for decimals in range(digits + 1)]


def run(model, duration=None):
    window = cb.Window(duration=duration) if duration is not None else None
    experiment = cb.Experiment(model, window=window) if window else cb.Experiment(model)
    return experiment.run(workers=1)


def value(results, model, name):
    return getattr(results[model], name).values[0, 0]


def number(steps, decimals):
    """``steps * 10**-decimals`` as the nearest float, from its exact decimal."""
    return float(f"{steps}e-{decimals}")


def shown(amount):
    """``amount`` as failure messages print it: ``%g`` with its shortest
    digits, and at least enough digits to write whole numbers in full."""
    digits = next(p for p in range(1, 18) if float(f"{amount:.{p - 1}e}") == amount)
    exponent = int(f"{amount:.{digits - 1}e}".split("e")[1])
    return f"{amount:.{max(digits, exponent + 1)}g}"


def half_away(amount, decimals):
    """The reference: the decimal Python prints, rounded half away from zero."""
    return float(Decimal(repr(amount)).quantize(Decimal(1).scaleb(-decimals),
                                                rounding=ROUND_HALF_UP))


# --- The scenario 0.7.6 fails ---------------------------------------------


class HundredMillion(cb.Model):
    tank: cb.FractionalContainer
    level_before: cb.Output[float]
    obtained: cb.Output[float]
    got_at: cb.Output[float]
    after: cb.Output[float]

    @cb.process
    def act(self):
        self.tank.put(100_000_000.0)            # receipt
        self.tank.put(0.3)                      # receipt
        self.tank.get(100_000_000.0)            # production issue
        self.level_before = self.tank.level()
        self.obtained = self.tank.get(0.3)      # sales order
        self.got_at = cb.now()
        self.after = self.tank.level()


def test_hundred_million_then_small_get_is_exact():
    model = HundredMillion()
    results = run(model, duration=5.0)
    assert not results.failed.any(), results.failure_reasons
    # 0.7.6 left 0.29999999701976776 here and the get(0.3) waited forever.
    assert value(results, model, "level_before") == 0.3
    assert value(results, model, "obtained") == 0.3
    assert value(results, model, "got_at") == 0.0
    assert value(results, model, "after") == 0.0


# --- Getting and waiting --------------------------------------------------


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
    results = run(model, duration=4.0)
    assert not results.failed.any(), results.failure_reasons
    assert value(results, model, "at_start") == 0.5
    # 0.5 at once, 0.25 at t=1 and t=2, the last 0.2 of the 0.25 put at t=3.
    assert value(results, model, "got_at") == 3.0
    # Exact, where a float level gives 0.04999999999999993.
    assert value(results, model, "after_get") == 0.05
    assert value(results, model, "at_end") == 0.05
    assert value(results, model, "peak") == 0.5
    # Level 0 over [0, 3), 0.05 over [3, 4].
    assert value(results, model, "mean") == pytest.approx(0.0125)
    times, values = results[model].tank.trial(0, 0)
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
    results = run(model)
    assert not results.failed.any(), results.failure_reasons
    # The first getter soaks up the first put; the second waits behind it
    # and gets its share of the leftover from the second put.
    assert value(results, model, "first_done") == 2.0
    assert value(results, model, "second_done") == 2.0


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


def test_interrupted_get_returns_what_it_took():
    model = Interrupted()
    results = run(model)
    assert not results.failed.any(), results.failure_reasons
    assert value(results, model, "woke_at") == 1.0
    assert value(results, model, "left") == 0.0
    assert value(results, model, "obtained") == 0.4


class Drifted(cb.Model):
    tank: cb.FractionalContainer
    obtained: cb.Output[float]
    got_at: cb.Output[float]
    after: cb.Output[float]

    @cb.process
    def act(self):
        self.tank.put(0.3)
        self.tank.get(0.1)
        self.tank.get(0.1)
        # In floats 0.3 - 0.1 - 0.1 is 0.09999999999999998, a hair under 0.1.
        self.obtained = self.tank.get(0.1)
        self.got_at = cb.now()
        self.after = self.tank.level()


def test_three_tenths_drain_exactly():
    model = Drifted()
    results = run(model, duration=1.0)
    assert not results.failed.any(), results.failure_reasons
    assert value(results, model, "got_at") == 0.0
    assert value(results, model, "obtained") == 0.1
    assert value(results, model, "after") == 0.0


class Shortfall(cb.Model):
    tank: cb.FractionalContainer
    need: cb.Param[float] = 0.0
    top_up: cb.Param[float] = 0.0
    got_at: cb.Output[float]

    @cb.process
    def consumer(self):
        self.tank.get(self.need)
        self.got_at = cb.now()

    @cb.process
    def filler(self):
        cb.hold(2.0)
        self.tank.put(self.top_up)


@pytest.mark.parametrize(("decimals", "digits", "initial", "need", "top_up"), [
    (3, 13, 0.099, 0.1, 0.001),                       # one step short
    (3, 13, 0.09, 0.1, 0.01),
    (7, 13, 0.0999999, 0.1, 1e-7),                    # one fine step short
    (0, 18, 899_999_990_000_000_000.0, 9e17, 1e10),   # near the 18-digit maximum
])
def test_a_real_shortfall_waits_for_the_put(decimals, digits, initial, need, top_up):
    model = Shortfall()
    model.tank = cb.FractionalContainer(initial=initial, decimals=decimals, digits=digits)
    model.need = need
    model.top_up = top_up
    results = run(model, duration=3.0)
    assert not results.failed.any(), results.failure_reasons
    assert value(results, model, "got_at") == 2.0


class ZeroGet(cb.Model):
    tank: cb.FractionalContainer
    got_at: cb.Output[float]

    @cb.process
    def act(self):
        self.tank.get(0.0)
        self.got_at = cb.now()


def test_get_zero_on_an_empty_tank_does_not_wait():
    model = ZeroGet()
    results = run(model, duration=1.0)
    assert not results.failed.any(), results.failure_reasons
    assert value(results, model, "got_at") == 0.0


class Huge(cb.Model):
    """Past 2**53 a float can't hold every integer, but the step count can:
    three puts of 3.333333333333333e+17 add exactly 999999999999999900."""

    tank: cb.FractionalContainer = cb.FractionalContainer(decimals=0, digits=18)
    total: cb.Output[float]
    got_at: cb.Output[float]

    @cb.process
    def act(self):
        third = 1e18 / 3.0
        self.tank.put(third)
        self.tank.put(third)
        self.tank.put(third)
        self.total = self.tank.get(999_999_999_999_999_900.0)
        self.got_at = cb.now()


def test_level_stays_exact_past_float_integers():
    model = Huge()
    results = run(model, duration=1.0)
    assert not results.failed.any(), results.failure_reasons
    assert value(results, model, "got_at") == 0.0
    assert value(results, model, "total") == float(999_999_999_999_999_900)


# --- Rounding at every precision ------------------------------------------


class Rounder(cb.Model):
    tank: cb.FractionalContainer
    amount: cb.Param[float] = 0.0
    rounded: cb.Output[float]

    @cb.process
    def act(self):
        self.rounded = self.tank.round(self.amount)


def rounded(amounts, decimals, digits=18):
    model = Rounder()
    model.tank = cb.FractionalContainer(decimals=decimals, digits=digits)
    model.amount = cb.sweep(*amounts)
    results = run(model)
    assert not results.failed.any(), results.failure_reasons
    return results[model].rounded.values[:, 0].tolist()


@pytest.mark.parametrize(("amount", "decimals", "expected"), [
    (2.675, 2, 2.68),          # 2.67499999... in binary; the decimal rounds up
    (1.0005, 3, 1.001),
    (0.4999999999999999, 0, 0.0),
    (1.25e-5, 5, 1e-5),
])
def test_round_examples(amount, decimals, expected):
    assert rounded([amount], decimals) == [expected]


@pytest.mark.parametrize("decimals", range(19))
def test_halfway_rounds_away_from_zero_at_every_precision(decimals):
    steps = (0, 1, 7, 49)
    # n.5 steps rounds up to n + 1; n.4 steps rounds down to n. Written as
    # decimals (n5e-(d+1)) so each is its own shortest form.
    halves = [float(f"{10 * n + 5}e-{decimals + 1}") for n in steps]
    assert rounded(halves, decimals) == [number(n + 1, decimals) for n in steps]
    below = [float(f"{10 * n + 4}e-{decimals + 1}") for n in steps]
    assert rounded(below, decimals) == [number(n, decimals) for n in steps]


@pytest.mark.parametrize("decimals", range(19))
def test_round_matches_decimal_reference_at_every_precision(decimals):
    rng = np.random.default_rng(decimals)
    # Stay inside 18 digits: below 10**(18 - decimals) whole units.
    top = min(10.0 ** (18 - decimals) - 1, 1e15)
    amounts = [0.0, *rng.uniform(0, top, 20), *rng.uniform(0, min(top, 1.0), 20),
               *(round(x, int(k)) for x, k in zip(rng.uniform(0, min(top, 1000.0), 20),
                                                   rng.integers(0, 7, 20)))]
    amounts = [float(x) for x in amounts]
    assert rounded(amounts, decimals) == [half_away(x, decimals) for x in amounts]


def test_round_is_correct_past_float_integers():
    # Steps above 2**53 must round once, not twice, when read back.
    amounts = [999_999_999_999_999_900.0, 2.0 ** 53 + 2, 123_456_789_012_345_678.0]
    assert rounded(amounts, 0) == [half_away(x, 0) for x in amounts]
    amounts = [127038367297864.33, 970890177237764.4, 649674669673830.6]
    assert rounded(amounts, 2) == [half_away(x, 2) for x in amounts]


class Amount(cb.Model):
    tank: cb.FractionalContainer
    amount: cb.Param[float] = 0.0
    level_after: cb.Output[float]

    @cb.process
    def act(self):
        self.tank.put(self.amount)
        self.level_after = self.tank.level()


def put(tank, amount):
    model = Amount()
    model.tank = tank
    model.amount = amount
    results = run(model)
    return results, value(results, model, "level_after")


@pytest.mark.parametrize(("amount", "level"), [(3 * 0.1, 0.3), (0.0005, 0.001), (0.0004, 0.0)])
def test_computed_amounts_round_half_away_from_zero(amount, level):
    results, after = put(cb.FractionalContainer(), amount)
    assert not results.failed.any(), results.failure_reasons
    assert after == level


class Cycle(cb.Model):
    """Three steps in, one step out three times: exact, never waits."""

    tank: cb.FractionalContainer
    three: cb.Param[float] = 0.0
    step: cb.Param[float] = 0.0
    left: cb.Output[float]
    got_at: cb.Output[float]

    @cb.process
    def act(self):
        self.tank.put(self.three)
        self.tank.get(self.step)
        self.tank.get(self.step)
        self.tank.get(self.step)
        self.got_at = cb.now()
        self.left = self.tank.level()


@pytest.mark.parametrize("decimals", range(19))
def test_put_then_get_is_exact_at_every_precision(decimals):
    model = Cycle()
    model.tank = cb.FractionalContainer(decimals=decimals, digits=18)
    model.three = number(3, decimals)
    model.step = number(1, decimals)
    results = run(model, duration=1.0)
    assert not results.failed.any(), results.failure_reasons
    assert value(results, model, "got_at") == 0.0
    assert value(results, model, "left") == 0.0


# --- Strict amounts ------------------------------------------------------


@pytest.mark.parametrize(("amount", "decimals"), [
    (3 * 0.1, 3), (0.5, 0), (0.25, 1), (1.0005, 3), (1e-18, 17), (1.5e-17, 17),
])
def test_strict_rejects_too_many_decimals(amount, decimals):
    results, _ = put(cb.FractionalContainer(decimals=decimals, digits=18, strict=True), amount)
    assert results.failed.all()
    assert results.failure_reasons[0][0] == (
        f"amount.tank: {shown(amount)} has too many decimals; "
        f"only {decimals} decimal(s) permitted")


@pytest.mark.parametrize(("amount", "decimals"), [
    (0.125, 3), (7.0, 0), (0.25, 2), (1e-18, 18), (1.5e-17, 18), (0.0, 0),
])
def test_strict_accepts_exact_amounts(amount, decimals):
    results, level = put(cb.FractionalContainer(decimals=decimals, digits=18, strict=True), amount)
    assert not results.failed.any(), results.failure_reasons
    assert level == amount


# --- Digit limits: the level fails exactly one step past 10**digits - 1 ---


class Filled(cb.Model):
    """Puts 9 * 10**k steps for k < digits, reaching exactly 10**digits - 1
    steps, then puts ``last``. Each piece is an exact int64 step count divided
    once by an exact power of ten, so it is the float nearest 9e(k - decimals),
    which a float states exactly even where 10**digits - 1 itself has more
    digits than a float can hold."""

    tank: cb.FractionalContainer
    digits: cb.Param[int] = 1
    scale: cb.Param[float] = 1.0
    last: cb.Param[float] = 0.0
    full: cb.Output[float]

    @cb.process
    def act(self):
        steps = 9
        for _ in range(self.digits):
            self.tank.put(steps / self.scale)
            steps *= 10
        self.full = self.tank.level()
        self.tank.put(self.last)


def fill(digits, decimals, last):
    model = Filled()
    model.tank = cb.FractionalContainer(decimals=decimals, digits=digits)
    model.digits = digits
    model.scale = float(10 ** decimals)
    model.last = last
    return model, run(model)


@pytest.mark.parametrize(("digits", "decimals"), PAIRS)
def test_level_fills_to_the_maximum_and_overflows_one_step_past(digits, decimals):
    model, results = fill(digits, decimals, 0.0)
    assert not results.failed.any(), results.failure_reasons
    assert value(results, model, "full") == number(10 ** digits - 1, decimals)
    _, results = fill(digits, decimals, number(1, decimals))
    assert results.failed.all()
    assert results.failure_reasons[0][0] == (
        f"filled.tank: {shown(number(1, decimals))} overflows the level past "
        f"the container's {digits} digit(s)")


@pytest.mark.parametrize(("digits", "decimals"), PAIRS)
def test_an_amount_past_the_digits_is_rejected(digits, decimals):
    # Exactly the maximum is accepted as a single amount...
    results, level = put(cb.FractionalContainer(decimals=decimals, digits=digits),
                         float(f"9e{digits - 1 - decimals}"))
    assert not results.failed.any(), results.failure_reasons
    # ...but 10**digits steps has one digit too many.
    too_large = float(f"1e{digits - decimals}")
    results, _ = put(cb.FractionalContainer(decimals=decimals, digits=digits), too_large)
    assert results.failed.all()
    assert results.failure_reasons[0][0] == (
        f"amount.tank: {shown(too_large)} exceeds the container's {digits} digit(s)")


@pytest.mark.parametrize(("digits", "decimals"), [(1, 0), (5, 5), (13, 3), (14, 0), (14, 14)])
def test_rounding_up_into_an_extra_digit_overflows(digits, decimals):
    # 10**digits - 0.5 steps (9.5, 99999.5, ...) rounds up to 10**digits steps:
    # one digit too many. Past 14 digits a float can't state the .5 exactly.
    amount = float(f"{10 ** (digits + 1) - 5}e-{decimals + 1}")
    results, _ = put(cb.FractionalContainer(decimals=decimals, digits=digits), amount)
    assert results.failed.all()
    assert results.failure_reasons[0][0] == (
        f"amount.tank: {shown(amount)} exceeds the container's {digits} digit(s)")
    # Half a step less rounds down to exactly the maximum and is accepted.
    below = float(f"{10 ** (digits + 1) - 6}e-{decimals + 1}")
    results, level = put(cb.FractionalContainer(decimals=decimals, digits=digits), below)
    assert not results.failed.any(), results.failure_reasons
    assert level == number(10 ** digits - 1, decimals)


@pytest.mark.parametrize("amount", [-1.0, -1e-9, math.nan, math.inf])
def test_invalid_amounts_fail_the_trial(amount):
    results, _ = put(cb.FractionalContainer(), amount)
    assert results.failed.all()


# --- Host-side configuration ----------------------------------------------


def test_defaults():
    tank = cb.FractionalContainer()
    assert (tank.decimals, tank.digits, tank.strict) == (3, 13, False)


def test_initial_must_be_finite_and_nonnegative():
    assert cb.FractionalContainer(initial=2).initial == 2.0
    assert cb.FractionalContainer(initial=np.float32(0.5)).initial == 0.5  # pyright: ignore[reportArgumentType]
    assert cb.FractionalContainer(initial=np.int64(3)).initial == 3.0  # pyright: ignore[reportArgumentType]
    for bad in (-0.5, math.nan, math.inf, True, "1"):
        with pytest.raises(ValueError):
            cb.FractionalContainer(initial=bad)  # pyright: ignore[reportArgumentType]


def test_initial_is_rounded_or_rejected():
    assert cb.FractionalContainer(initial=0.1 + 0.2).initial_steps == 300
    with pytest.raises(ValueError, match="too many decimals"):
        cb.FractionalContainer(initial=0.1 + 0.2, strict=True)


@pytest.mark.parametrize(("digits", "decimals"), PAIRS)
def test_initial_accepts_the_maximum_and_rejects_one_step_more(digits, decimals):
    # The largest initial a float states exactly: 9 then zeros, 10**digits - 1
    # itself needs digits significant figures, more than a float has past 15.
    top = float(f"9e{digits - 1 - decimals}")
    assert cb.FractionalContainer(initial=top, decimals=decimals,
                                  digits=digits).initial_steps == 9 * 10 ** (digits - 1)
    with pytest.raises(ValueError, match=f"exceeds {digits} digit"):
        cb.FractionalContainer(initial=float(f"1e{digits - decimals}"),
                               decimals=decimals, digits=digits)


@pytest.mark.parametrize("settings", [
    {"decimals": -1}, {"decimals": 14}, {"decimals": 1.5}, {"decimals": True},
    {"decimals": 3, "digits": 0}, {"decimals": 3, "digits": 19},
    {"decimals": 3, "digits": 2}, {"digits": True},
])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        cb.FractionalContainer(**settings)


def test_container_rejects_float_amounts():
    class Truncating(cb.Model):
        stock: cb.Container

        @cb.process
        def act(self):
            self.stock.put(0.7)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ModelCompileError, match="FractionalContainer"):
        cb.Experiment(Truncating()).run(workers=1)
