"""Check observable evaluation semantics, independent of generated source."""
import numpy as np
import cimba.sim as sim
from cimba import random, _bindings as native


class Item(sim.Component):
    value: sim.Param

    def __init__(self, value):
        self.value = value

    @sim.function
    def read(self, offset: float) -> float:
        return self.value + offset


class Twice(Item):
    @sim.function
    def read(self, offset: float) -> float:
        return self.value * 2 + offset


class Thrice(Item):
    @sim.function
    def read(self, offset: float) -> float:
        return self.value * 3 + offset


class Semantics(sim.Model):
    items: list[Item] = [Item(10), Twice(20), Thrice(30)]
    expected_argument: sim.Output
    actual_argument: sim.Output
    expected_receiver: sim.Output
    actual_receiver: sim.Output
    local_index: sim.Output
    conditional: sim.Output

    @sim.function
    def by_index(self, index: int) -> float:
        return self.items[index].value

    @sim.function
    def next_index(self, index: int) -> float:
        index += 1
        return self.items[index].read(0.0)

    @sim.function
    def choose(self, enabled: bool) -> float:
        if enabled:
            return self.items[random.dice(0, 2)].read(0.0)
        return 7.0

    @sim.process
    def compare(self):
        native.random_initialize(self.seed)
        index = random.dice(0, 2)
        self.expected_argument = self.items[index].value
        native.random_initialize(self.seed)
        self.actual_argument = self.by_index(random.dice(0, 2))
        native.random_initialize(self.seed)
        index = random.dice(0, 2)
        offset = random.uniform(0.0, 1.0)
        self.expected_receiver = 10.0 * (index + 1) ** 2 + offset
        native.random_initialize(self.seed)
        self.actual_receiver = self.items[random.dice(0, 2)].read(
            random.uniform(0.0, 1.0))
        self.local_index = self.next_index(0)
        native.random_initialize(self.seed)
        expected = random.uniform(0.0, 1.0)
        native.random_initialize(self.seed)
        self.conditional = self.choose(False) + random.uniform(0.0, 1.0) - expected


def test_arguments_receivers_and_conditional_reads_preserve_evaluation_order():
    experiment = Semantics().experiment(replications=32, duration=None, warmup=0, seed=9)
    assert experiment.run() == 0
    np.testing.assert_array_equal(experiment["actual_argument"], experiment["expected_argument"])
    np.testing.assert_array_equal(experiment["actual_receiver"], experiment["expected_receiver"])
    np.testing.assert_array_equal(experiment["local_index"], 40.0)
    np.testing.assert_allclose(experiment["conditional"], 7.0, rtol=0, atol=2e-15)
