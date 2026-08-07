"""Pyright fixture for typed experiment result schemas.

Run with ``uv run pyright``.
"""

from typing import Protocol, assert_type

import numpy as np
from numpy.typing import NDArray

import cimba.sim as sim


class CounterResults(Protocol):
    mean_queue_length: NDArray[np.float64]
    line: tuple[tuple[NDArray[np.float64], ...], ...]


class QueueResults(Protocol):
    customers_served: NDArray[np.float64]
    waits: tuple[NDArray[np.float64], ...]
    counters: CounterResults


class Counter(sim.Component):
    mean_queue_length: sim.Output
    line: sim.Queue


class QueueModel(sim.Model[QueueResults]):
    counters: list[Counter] = [Counter()]
    customers_served: sim.Output
    waits: sim.Dataset
    q: sim.Queue


model = QueueModel()
experiment = model.experiment()

assert_type(experiment.results, QueueResults)
assert_type(experiment.results.customers_served,
            NDArray[np.float64])
assert_type(experiment.results.counters.mean_queue_length,
            NDArray[np.float64])
assert_type(experiment.results.waits,
            tuple[NDArray[np.float64], ...])
assert_type(experiment.results.counters.line,
            tuple[tuple[NDArray[np.float64], ...], ...])
