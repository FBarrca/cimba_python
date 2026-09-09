"""Random streams stay consistent across the compiled callback boundaries."""

import numpy as np
from numba import njit

import cimba
import cimba.random as random_alias
import cimba.sim as sim
from cimba.random import normal as normal_draw


@njit
def gamma_draw():
    return random_alias.gamma(shape=2.0)


@njit
def expected_draws():
    return (
        cimba.random.exponential(mean=2.0),
        normal_draw(sigma=0.5),
        gamma_draw(),
        random_alias.uniform(max=3.0),
        cimba.random.beta(a=2.0, b=3.0),
    )


class Sampler(sim.Component):
    mean: sim.Param = 2.0
    value: sim.Output

    @sim.function
    def sample(self) -> float:
        return cimba.random.exponential(mean=self.mean)

    @sim.process
    def draw(self, env):
        self.value = self.sample()
        env.normal_value = normal_draw(sigma=0.5)
        env.gamma_value = gamma_draw()
        env.alarm.schedule(0.0)


class RandomCallbacks(sim.Model):
    sampler: Sampler = Sampler()
    normal_value: sim.Output
    gamma_value: sim.Output
    event_value: sim.Output
    collected_value: sim.Output
    alarm: sim.Event

    @sim.event(field="alarm")
    def on_alarm(self):
        self.event_value = random_alias.uniform(max=3.0)

    @sim.collect
    def collect_draw(self):
        self.collected_value = cimba.random.beta(a=2.0, b=3.0)


class ReferenceDraws(sim.Model):
    first: sim.Output
    second: sim.Output
    third: sim.Output
    fourth: sim.Output
    fifth: sim.Output

    @sim.process
    def draw(self):
        values = expected_draws()
        self.first = values[0]
        self.second = values[1]
        self.third = values[2]
        self.fourth = values[3]
        self.fifth = values[4]


def test_callbacks_and_njit_helpers_share_the_trial_random_stream():
    experiment = RandomCallbacks().experiment(
        replications=16, duration=1.0, warmup=0.0, seed=42)
    reference = ReferenceDraws().experiment(
        replications=16, duration=1.0, warmup=0.0, seed=42)
    np.testing.assert_array_equal(experiment.trials["seed"], reference.trials["seed"])
    assert reference.run() == 0
    expected = np.column_stack([
        reference[field] for field in ("first", "second", "third", "fourth", "fifth")])

    fields = (
        "sampler__value", "normal_value", "gamma_value",
        "event_value", "collected_value",
    )
    for _ in range(2):
        assert experiment.run() == 0
        actual = np.column_stack([experiment[field] for field in fields])
        np.testing.assert_array_equal(actual, expected)
