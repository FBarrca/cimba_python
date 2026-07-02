import pytest

import cimba.sim as sim


class Warehouse(sim.Component):
    on_hand: sim.State
    ordered: sim.Output
    orders: sim.Queue

    def __init__(self, R: int, B: int):
        self.R = R
        self.B = B

    @sim.process
    def reorder(self, env):
        if self.on_hand < self.R:
            sim.put(self.orders, self.B)
            self.on_hand += self.B
            self.ordered = 1.0


def test_component_declarations_flatten_into_model_dtype():
    class Bin(sim.Component):
        cap: sim.Param
        done: sim.Output
        level: sim.State
        queue: sim.Queue = sim.capacity("cap")
        lanes: sim.PQueues = sim.count(2)

    class Network(sim.Model):
        left: Bin = Bin()

    model = Network()
    assert model.params == ["left__cap"]
    assert model.outputs == ["left__done"]
    assert model.state == ["left__level"]
    assert model.queues == {"left__queue": "left__cap"}
    assert model.pqueues == {"left__lanes": 2}
    for field in ("left__cap", "left__done", "left__level",
                  "left__queue", "left__lanes"):
        assert field in model.dtype.fields


def test_component_process_runs_with_fields_and_constants():
    class Network(sim.Model):
        retailer: Warehouse = Warehouse(R=20, B=50)

    model = Network()
    assert [p.name for p in model._processes] == ["retailer__reorder"]
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=12)
    assert exp.run() == 0
    assert exp["retailer__ordered"][0] == 1.0
    assert exp.trials["retailer__on_hand"][0] == 50


def test_component_process_copies_and_priority_are_registered():
    class Workers(sim.Component):
        count: sim.State

        @sim.process(copies=3, priority=7)
        def worker(self, env, idx):
            self.count += idx + 1

    class Network(sim.Model):
        workers: Workers = Workers()

    model = Network()
    (proc,) = model._processes
    assert proc.name == "workers__worker"
    assert proc.copies == 3
    assert proc.priority == 7
    assert proc.indexed


def test_component_process_dag_uses_lowered_source():
    class Network(sim.Model):
        retailer: Warehouse = Warehouse(R=20, B=50)

    graph = Network().process_dag()
    nodes = {node.key for node in graph.nodes}
    edges = {(edge.source, edge.target, edge.label) for edge in graph.edges}
    assert "process:retailer__reorder" in nodes
    assert "queue:retailer__orders" in nodes
    assert "state:retailer__on_hand" in nodes
    assert ("process:retailer__reorder", "queue:retailer__orders",
            "put") in edges
    assert ("state:retailer__on_hand", "process:retailer__reorder",
            "read") in edges


def test_model_collect_can_use_component_field_namespace():
    class Station(sim.Component):
        queue: sim.Queue

    class Network(sim.Model):
        avg: sim.Output
        station: Station = Station()

    model = Network()

    @model.process
    def feeder(env: Network):
        sim.put(env.station.queue, 2)
        sim.suspend()

    @model.collect
    def collect_stats(env: Network):
        env.avg = sim.mean_level(env.station.queue)

    assert "env.station__queue" in model._processes[0].fn.__cimba_source__
    assert "env.station__queue" in model._collect.__cimba_source__
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=13)
    assert exp.run() == 0
    assert exp["avg"][0] == 2.0


def test_model_process_can_read_and_write_component_state_namespace():
    class Counter(sim.Component):
        count: sim.State

    class Network(sim.Model):
        done: sim.Output
        counter: Counter = Counter()

    model = Network()

    @model.process
    def actor(env: Network):
        env.counter.count += 3
        env.done = env.counter.count
        sim.suspend()

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=14)
    assert exp.run() == 0
    assert exp["done"][0] == 3.0
    assert exp.trials["counter__count"][0] == 3


def test_model_predicate_can_use_component_field_namespace():
    class GateState(sim.Component):
        open: sim.State

    class Network(sim.Model):
        ok: sim.Output
        gate: sim.Condition
        ready: sim.Predicate
        flags: GateState = GateState()

    model = Network()

    @model.predicate
    def ready(env: Network) -> bool:
        return env.flags.open == 1

    @model.process
    def opener(env: Network):
        sim.hold(1.0)
        env.flags.open = 1
        sim.signal(env.gate)
        sim.suspend()

    @model.process
    def waiter(env: Network):
        sim.wait_for(env.gate, env.ready, env)
        env.ok = 1.0
        sim.suspend()

    exp = model.experiment(replications=1, duration=2.0, warmup=0.0,
                           seed=15)
    assert exp.run() == 0
    assert exp["ok"][0] == 1.0


def test_model_event_can_use_component_field_namespace():
    class Counter(sim.Component):
        count: sim.State

    class Network(sim.Model):
        bump: sim.Event
        counter: Counter = Counter()

    model = Network()

    @model.event
    def bump(env: Network):
        env.counter.count += 1

    @model.process
    def driver(env: Network):
        sim.schedule(env.bump, env, 1.0)
        sim.hold(2.0)
        sim.suspend()

    exp = model.experiment(replications=1, duration=3.0, warmup=0.0,
                           seed=16)
    assert exp.run() == 0
    assert exp.trials["counter__count"][0] == 1


def test_model_process_dag_uses_lowered_component_namespace_source():
    class Station(sim.Component):
        queue: sim.Queue
        count: sim.State

    class Network(sim.Model):
        station: Station = Station()

    model = Network()

    @model.process
    def feeder(env: Network):
        env.station.count += 1
        sim.put(env.station.queue, 1)
        sim.suspend()

    graph = model.process_dag()
    nodes = {node.key for node in graph.nodes}
    edges = {(edge.source, edge.target, edge.label) for edge in graph.edges}
    assert "queue:station__queue" in nodes
    assert "state:station__count" in nodes
    assert ("process:feeder", "queue:station__queue", "put") in edges
    assert ("process:feeder", "state:station__count", "write") in edges
    assert ("state:station__count", "process:feeder", "read") in edges


def test_component_duplicate_flattened_name_is_rejected():
    class Bin(sim.Component):
        queue: sim.Queue

    class Network(sim.Model):
        left__queue: sim.Queue
        left: Bin = Bin()

    with pytest.raises(ValueError, match="duplicate field name 'left__queue'"):
        Network()


def test_component_missing_and_wrong_defaults_are_rejected():
    class Bin(sim.Component):
        queue: sim.Queue

    class Missing(sim.Model):
        left: Bin

    with pytest.raises(ValueError, match="needs a Bin instance default"):
        Missing()

    class Wrong(sim.Model):
        left: Bin = object()

    with pytest.raises(TypeError, match="default must be a Bin instance"):
        Wrong()


def test_component_unsupported_self_usage_is_rejected():
    class Dynamic(sim.Component):
        value: sim.State

        @sim.process
        def actor(self, env):
            self.value = getattr(self, "value")

    class CallsMethod(sim.Component):
        value: sim.State

        def helper(self):
            return 1

        @sim.process
        def actor(self, env):
            self.value = self.helper()

    class UsesObjectConstant(sim.Component):
        value: sim.State

        def __init__(self):
            self.values = [1]

        @sim.process
        def actor(self, env):
            self.value = self.values[0]

    class DynamicModel(sim.Model):
        dyn: Dynamic = Dynamic()

    with pytest.raises(ValueError, match="dynamic getattr"):
        DynamicModel()

    class CallsModel(sim.Model):
        calls: CallsMethod = CallsMethod()

    with pytest.raises(ValueError, match="cannot call self.helper"):
        CallsModel()

    class ConstantModel(sim.Model):
        constants: UsesObjectConstant = UsesObjectConstant()

    with pytest.raises(ValueError, match="unsupported self.values"):
        ConstantModel()


def test_component_capacity_must_reference_component_param_if_local():
    class Bin(sim.Component):
        cap: sim.State
        queue: sim.Queue = sim.capacity("cap")

    class Network(sim.Model):
        bin: Bin = Bin()

    with pytest.raises(ValueError, match="capacity 'cap' must name a Param"):
        Network()


def test_model_component_namespace_errors_are_rejected():
    class Box(sim.Component):
        value: sim.State

    class UsesBox(sim.Model):
        box: Box = Box()

    direct = UsesBox()
    with pytest.raises(ValueError, match="cannot use env.box directly"):
        @direct.process
        def direct_actor(env: UsesBox):
            _ = env.box
            sim.suspend()

    unknown = UsesBox()
    with pytest.raises(ValueError, match="unknown component field"):
        @unknown.process
        def unknown_actor(env: UsesBox):
            env.box.missing = 1
            sim.suspend()

    dynamic = UsesBox()
    with pytest.raises(ValueError, match="dynamic getattr"):
        @dynamic.process
        def dynamic_actor(env: UsesBox):
            env.box.value = getattr(env.box, "value")
            sim.suspend()

    assign = UsesBox()
    with pytest.raises(ValueError, match="cannot use env.box directly"):
        @assign.process
        def assign_actor(env: UsesBox):
            env.box = 1
            sim.suspend()

    nested = UsesBox()
    with pytest.raises(ValueError, match="below component field"):
        @nested.process
        def nested_actor(env: UsesBox):
            _ = env.box.value.extra
            sim.suspend()
