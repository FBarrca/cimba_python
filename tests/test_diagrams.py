"""Structure and process diagrams are inferred without compiling models."""

import numba
import pytest

import cimba as cb
from cimba import inputs
from cimba.diagrams import Edge, mermaid, process_graph, structure


class MM1(cb.Model):
    interarrival: cb.Input[float] = inputs.dist.exponential(mean=2.0)
    service: cb.Input[float]
    queue: cb.Container
    served: cb.Dataset

    @cb.process
    def arrivals(self):
        while True:
            cb.hold(self.interarrival.next())
            self.queue.put(1)

    @cb.process
    def server(self):
        while True:
            self.queue.get(1)
            cb.hold(self.service.next())
            self.served.record(cb.now())


def test_process_graph_mermaid_and_dot_are_stable():
    graph = process_graph(MM1())
    assert graph.to_mermaid() == "\n".join([
        "flowchart TD",
        '  subgraph g0["mm1: MM1"]',
        '    n0[/"interarrival · exponential"/]',
        '    n1[/"service · no source"/]',
        '    n2["queue · Container"]',
        '    n3["served · Dataset"]',
        '    n4(["arrivals"])',
        '    n5(["server"])',
        "  end",
        "  n0 -->|next| n4",
        "  n4 -->|put| n2",
        "  n2 -->|get| n5",
        "  n1 -->|next| n5",
        "  n5 -->|record| n3",
    ])
    dot = graph.to_dot()
    assert dot.startswith("digraph cimba {")
    assert '"mm1.queue" -> "mm1.server" [label="get"];' in dot
    assert "subgraph cluster_0" in dot


def test_graph_order_and_cycles():
    graph = process_graph(MM1())
    order = graph.topological_order()
    assert order.index("mm1.arrivals") < order.index("mm1.queue") < order.index("mm1.server")
    cyclic = type(graph)(graph.nodes, graph.edges + (Edge("mm1.server", "mm1.arrivals"),))
    with pytest.raises(ValueError, match="cycle"):
        cyclic.topological_order()


@numba.njit
def _take(station):
    station.buffer.get(1)


class Station(cb.Model):
    buffer: cb.Container
    upstream: cb.Ref["Station"] | None = None

    @cb.process
    def work(self):
        _take(self)
        if self.upstream is not None:
            target = self.upstream.buffer
            target.put(1)


class Line(cb.Model):
    stations: list[Station]
    first_only: cb.Container

    def __init__(self):
        first = Station()
        self.stations = [first, Station(), Station()]
        self.stations[1].upstream = first

    @cb.process
    def feed(self):
        self.stations[0].buffer.put(1)
        for i in range(len(self.stations)):
            self.stations[i].buffer.put(1)


def test_aliases_helpers_references_and_list_indices():
    graph = process_graph(Line())
    edges = {(e.source, e.label, e.target) for e in graph.edges}
    # njit helper taking the model view
    assert ("line.stations[0].buffer", "get", "line.stations[0].work") in edges
    # alias of an entity reached through a reference
    assert ("line.stations[1].work", "put", "line.stations[0].buffer") in edges
    # literal index hits one item, non-literal index hits every item
    assert {("line.feed", "put", f"line.stations[{i}].buffer") for i in range(3)} <= edges
    # a None reference contributes nothing
    assert not any(e[0] == "line.stations[2].work" and e[1] == "put" for e in edges)


class Job(cb.Model):
    shop: cb.Ref["Shop"]
    size: cb.Input[float]
    done: cb.State[float] = 0.0

    @cb.process
    def run(self):
        self.shop.machine.acquire(1)
        cb.hold(self.size.next())
        self.shop.machine.release(1)
        self.shop.finished.put(self)


class Shop(cb.Model):
    sizes: cb.Input[float] = inputs.dist.gamma(shape=2.0)
    machine: cb.Resource = cb.Resource(capacity=2)
    finished: cb.Store[Job]
    gate: cb.Condition
    open: cb.State[bool] = False

    @cb.predicate
    def is_open(self):
        return self.open

    @cb.event
    def opening(self):
        self.open = True
        self.gate.signal()

    @cb.process
    def jobs(self):
        cb.schedule(self.opening, 8.0)
        self.gate.wait_until(self.is_open)
        while True:
            cb.hold(1.0)
            cb.spawn(Job, shop=self, size=self.sizes)

    @cb.process
    def sink(self):
        while True:
            job = self.finished.get()
            job.done = cb.now()
            cb.release(job)

    @cb.on_end
    def report(self):
        self.open = False


def test_spawned_models_events_conditions_state_and_hooks():
    shop = Shop()
    edges = {(e.source, e.label, e.target) for e in process_graph(shop).edges}
    assert ("shop.jobs", "spawn", "Job") in edges
    assert ("shop.machine", "acquire", "Job.run") in edges      # via spawn(shop=self)
    assert ("shop.sizes", "next", "Job.run") in edges           # spawned input from parent
    assert ("Job.run", "put", "shop.finished") in edges
    assert ("shop.jobs", "schedule", "shop.opening") in edges
    assert ("shop.opening", "signal", "shop.gate") in edges
    assert ("shop.gate", "wait", "shop.jobs") in edges
    assert not any(label == "write" for _, label, _ in edges)

    detailed = process_graph(shop, state=True, hooks=True)
    edges = {(e.source, e.label, e.target) for e in detailed.edges}
    assert ("shop.sink", "write", "Job.done") in edges          # model from a typed store
    assert ("shop.report", "write", "shop.open") in edges
    assert detailed.node("shop.report").kind == "hook"
    assert detailed.node("Job").label == "new Job"


def test_structure_graph_matches_mermaid_and_needs_no_sources():
    line = Line()
    graph = structure(line)
    assert graph.to_mermaid() == mermaid(line)
    assert ("line", "line.stations[1]", "stations[1]", "solid") in {
        (e.source, e.target, e.label, e.style) for e in graph.edges}
    assert ("line.stations[1]", "line.stations[0]", "upstream", "dotted") in {
        (e.source, e.target, e.label, e.style) for e in graph.edges}
    # MM1.service has no source yet: drawing still works, running does not.
    process_graph(MM1())
    with pytest.raises(ValueError, match="source missing"):
        cb.Experiment(MM1())


@pytest.mark.parametrize("module, factory, expected", [
    ("tutorial.tut_1_1", "MM1", ("mm1.queue", "get", "mm1.service")),
    ("tutorial.tut_4_1", "Harbor", ("harbor.facilities.tugs", "acquire", "Ship.voyage")),
    ("tutorial.tut_3_1", "Park", ("Visitor.visit", "enqueue", "park.ride_queues[0].line")),
    ("tutorial.tut_5_1", "AssemblyLine",
     ("assemblyline.station_1.server", "put", "assemblyline.station_2.inbox")),
    ("tutorial.multi_echelon_inventory", "MultiEchelonInventory",
     ("multiecheloninventory.facilities[1].place_order", "put",
      "multiecheloninventory.facilities[0].orders")),
])
def test_tutorial_process_graphs(module, factory, expected):
    from importlib import import_module
    model = getattr(import_module(module), factory)()
    graph = process_graph(model)
    assert expected in {(e.source, e.label, e.target) for e in graph.edges}
    graph.to_mermaid()
    graph.to_dot()
