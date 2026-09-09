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


def test_callbacks_share_the_standalone_njit_random_stream():
    experiment = RandomCallbacks().experiment(
        replications=16, duration=1.0, warmup=0.0, seed=42)
    expected = []
    for seed in experiment.trials["seed"]:
        cimba.random.seed(int(seed))
        expected.append(expected_draws())

    fields = (
        "sampler__value", "normal_value", "gamma_value",
        "event_value", "collected_value",
    )
    for _ in range(2):
        assert experiment.run() == 0
        actual = np.column_stack([experiment[field] for field in fields])
        np.testing.assert_array_equal(actual, expected)
