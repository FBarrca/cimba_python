"""Structural design points: sweeps of references and collections, resolved
on each design point's own model tree (issue #29)."""

import numpy as np
import pytest

import cimba as cb
from cimba import inputs
from cimba.schema import Assembly, ModelDefinitionError


class Node(cb.Model):
    delay: cb.Param[float] = 1.0
    fired: int = 0
    count: cb.Output[int]

    @cb.process
    def run(self):
        while True:
            cb.hold(self.delay)
            self.fired += 1

    @cb.on_end
    def measure(self):
        self.count = self.fired


class Graph(cb.Model):
    entry: Node


class GraphA(Graph):
    def __init__(self):
        self.entry = Node()
        self.entry.delay = 1.0


class GraphB(Graph):
    review: Node

    def __init__(self):
        self.entry = Node()
        self.entry.delay = 2.0
        self.review = Node()
        self.review.delay = 3.0


class Driver(cb.Model):
    """A stable model outside the swept child that references a node."""
    watch: cb.Ref[Node]
    seen: cb.Output[float]
    draw: cb.Output[float]
    noise: cb.Input[float] = inputs.dist.exponential(mean=1.0)

    @cb.process
    def run(self):
        self.draw = self.noise.next()
        self.seen = self.watch.delay


class Root(cb.Model):
    graph: Graph
    driver: Driver


def window():
    return cb.Window(duration=10.0)


def test_a_reference_linked_to_a_swept_child_follows_its_option():
    root = Root()
    a, b = GraphA(), GraphB()
    root.driver = Driver()
    graphs, entries = cb.sweeps([a, b], [a.entry, b.entry])
    root.graph = graphs
    root.driver.watch = entries
    results = cb.Experiment(root, replications=2, window=window(), seed=1).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results.meta.variants == 2
    assert results[root.driver].seen.values.tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert np.isnan(results[b.review].count.values[0]).all()
    assert (results[b.review].count.values[1] == 3).all()
    draws = results[root.driver].draw.values
    assert (draws[0] == draws[1]).all()                      # common random numbers


def test_a_reference_into_an_unselected_option_fails_at_construction():
    root = Root()
    a, b = GraphA(), GraphB()
    root.graph = cb.sweep(a, b)
    root.driver = Driver()
    root.driver.watch = a.entry
    with pytest.raises(ModelDefinitionError,
                       match=r"design point 1 \(root\.graph#1\): root\.driver\.watch: "
                             r"dangling reference"):
        cb.Experiment(root, window=window())


class Two(cb.Model):
    fast: Node
    slow: Node
    driver: Driver


def two():
    model = Two()
    model.fast, model.slow, model.driver = Node(), Node(), Driver()
    model.fast.delay, model.slow.delay = 1.0, 4.0
    return model


def test_a_swept_reference_is_an_independent_axis():
    model = two()
    model.driver.watch = cb.sweep(model.fast, model.slow)
    model.fast.delay = cb.sweep(0.5, 1.0)
    experiment = cb.Experiment(model, replications=2, window=window(), seed=4)
    assert len(experiment.design.points) == 4
    results = experiment.run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    seen = results[model.driver].seen.values[:, 0].tolist()
    targets = results.levels(model.driver.watch)
    delays = results.levels(model.fast.delay)
    assert seen == [delay if target is model.fast else 4.0
                    for target, delay in zip(targets, delays)]
    draws = results[model.driver].draw.values
    assert (draws == draws[0]).all()
    # Every node exists in every design point, so none is reported absent
    assert not np.isnan(results[model.slow].count.values).any()
    again = experiment.only(trials=[7])
    assert again[model.driver].seen[0, 0] == results[model.driver].seen[3, 1]


def test_a_reference_option_is_checked_against_the_tree():
    model = two()
    stray = Node()
    model.driver.watch = cb.sweep(model.fast, stray)
    with pytest.raises(ModelDefinitionError, match=r"two\.driver\.watch#1: dangling reference"):
        cb.Experiment(model)


class Hub(cb.Model):
    outgoing: list[cb.Ref[Node]]
    width: cb.Output[int]
    total: cb.Output[float]

    @cb.process
    def run(self):
        self.width = len(self.outgoing)
        total = 0.0
        for i in range(len(self.outgoing)):
            total += self.outgoing[i].delay
        self.total = total


class Fan(cb.Model):
    a: Node
    b: Node
    hub: Hub


def test_a_swept_list_of_references():
    model = Fan()
    model.a, model.b, model.hub = Node(), Node(), Hub()
    model.b.delay = 2.0
    model.hub.outgoing = cb.sweep([model.a], [model.a, model.b])
    results = cb.Experiment(model, window=window()).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model.hub].width.values.ravel().tolist() == [1, 2]
    assert results[model.hub].total.values.ravel().tolist() == [1.0, 3.0]


class Pool(cb.Model):
    nodes: list[Node]
    width: cb.Output[int]

    @cb.process
    def run(self):
        self.width = len(self.nodes)


def test_a_swept_owned_collection_holds_only_its_option_items():
    pool = Pool()
    shared, extra = Node(), Node()
    pool.nodes = cb.sweep([shared], [shared, extra])
    labels = [instance.label for instance in Assembly.of(pool).instances]
    assert labels == ["pool", "pool.nodes#0[0]", "pool.nodes#1[1]"]
    results = cb.Experiment(pool, window=window()).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[pool].width.values.ravel().tolist() == [1, 2]
    assert results[shared].count.values.ravel().tolist() == [9, 9]
    counts = results[extra].count.values.ravel()
    assert np.isnan(counts[0]) and counts[1] == 9


def test_collection_sweeps_need_lists():
    pool = Pool()
    with pytest.raises(TypeError, match="sweep must contain lists of Node models"):
        pool.nodes = cb.sweep([Node(), Node()])   # two options, not two lists
    hub = Hub()
    with pytest.raises(TypeError, match="sweep must contain lists of references"):
        hub.outgoing = cb.sweep(Node(), Node())


class Plain(cb.Model):
    level: int = 0


def test_plain_fields_still_reject_sweeps():
    with pytest.raises(TypeError, match="is not a Param, input"):
        Plain().level = cb.sweep(1, 2)


class Crews(cb.Model):
    day: cb.Resource = cb.Resource(capacity=1)
    night: cb.Resource = cb.Resource(capacity=3)
    crew: cb.Ref[cb.Resource]
    pool: list[cb.Ref[cb.Resource]]
    capacity: cb.Output[int]
    pooled: cb.Output[int]

    @cb.process
    def run(self):
        self.capacity = self.crew.capacity()
        self.pooled = len(self.pool)


def test_entity_references_can_be_swept():
    model = Crews()
    model.crew = cb.sweep(model.day, model.night)
    model.pool = cb.sweep([model.day], [model.day, model.night])
    results = cb.Experiment(model).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].capacity.values.ravel().tolist() == [1, 1, 3, 3]
    assert results[model].pooled.values.ravel().tolist() == [1, 2, 1, 2]


def test_structure_draws_every_reference_option():
    from cimba.diagrams import structure

    model = two()
    model.driver.watch = cb.sweep(model.fast, model.slow)
    edges = {(e.source, e.target, e.label) for e in structure(model).edges}
    assert ("two.driver", "two.fast", "watch (option 0)") in edges
    assert ("two.driver", "two.slow", "watch (option 1)") in edges
