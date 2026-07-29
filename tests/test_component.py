import numpy as np
import pytest

import cimba.sim as sim


def test_nested_component_uses_concrete_default_function_override():
    class Policy(sim.Component):
        @sim.function
        def calculate(self, value: float) -> float:
            return value

    class DoublePolicy(Policy):
        @sim.function
        def calculate(self, value: float) -> float:
            return value * 2.0

    class Holder(sim.Component):
        policy: Policy = DoublePolicy()
        result: sim.Output

        @sim.collect
        def report(self, env):
            self.result = self.policy.calculate(3.0)

    class Example(sim.Model):
        holder: Holder = Holder()

    model = Example()

    @model.process
    def run(env: Example):
        sim.suspend()

    exp = model.experiment(replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp["holder__result"][0] == 6.0


def test_heterogeneous_component_collection_dispatches_dynamic_function_calls():
    class Policy(sim.Component):
        @sim.function
        def calculate(self, value: float) -> float:
            return value

    class FixedLot(Policy):
        lot_size: sim.Param = 10.0

        @sim.function
        def calculate(self, value: float) -> float:
            return value + self.lot_size

    class EOQ(Policy):
        ordering_cost: sim.Param = 4.0

        @sim.function
        def calculate(self, value: float) -> float:
            return value * self.ordering_cost

    class System(sim.Model):
        policies: list[Policy] = [FixedLot(), EOQ(), FixedLot()]
        total: sim.Output

    model = System()
    assert model.dtype["policies__lot_size"].shape == (2,)
    assert model.dtype["policies__ordering_cost"].shape == ()
    calculate_specs = [
        spec for spec in model._component_functions.values()
        if spec.decl.name == "policies" and spec.name == "calculate"
    ]
    assert len(calculate_specs) == 2
    assert {spec.instance_indices for spec in calculate_specs} == {
        (0, 2),
        (1,),
    }
    lot_schema = model.component_schema("policies.lot_size")
    assert isinstance(lot_schema, sim.ComponentFieldSchema)
    assert lot_schema.path == "policies[].lot_size"
    assert lot_schema.flattened_name == "policies__lot_size"
    assert lot_schema.kind == "param"
    assert lot_schema.owners == (0, 2)
    assert lot_schema.concrete_types == (FixedLot, FixedLot)
    assert lot_schema.logical_count == 3
    assert lot_schema.shape == (2,)
    assert lot_schema.packed

    @model.process
    def run(env: System):
        env.total = 0.0
        for i in range(3):
            env.total += env.policies[i].calculate(2.0)
        sim.suspend()

    exp = model.experiment(
        policies__lot_size={0: 3.0, 2: 5.0},
        policies__ordering_cost={1: 7.0},
        replications=1,
        duration=1.0,
    )
    assert exp.run() == 0
    assert exp["total"][0] == 26.0
    assert exp.trials["policies__lot_size"][0].tolist() == [3.0, 5.0]
    assert exp.trials["policies__ordering_cost"][0] == 7.0


def test_component_collection_can_have_different_nested_policy_types():
    class Policy(sim.Component):
        @sim.function
        def calculate(self, value: float) -> float:
            return value

    class DoublePolicy(Policy):
        @sim.function
        def calculate(self, value: float) -> float:
            return value * 2.0

    class TriplePolicy(Policy):
        @sim.function
        def calculate(self, value: float) -> float:
            return value * 3.0

    class Material(sim.Component):
        policy: Policy
        result: sim.Output

        def __init__(self, policy):
            self.policy = policy

        @sim.function
        def value(self, amount: float) -> float:
            return self.policy.calculate(amount)

        @sim.collect
        def report(self, env):
            self.result = self.value(3.0)

    class SupplyModel(sim.Model):
        materials: list[Material] = [
            Material(DoublePolicy()),
            Material(TriplePolicy()),
        ]
        dynamic_total: sim.Output

    model = SupplyModel()

    @model.process
    def run(env: SupplyModel):
        env.dynamic_total = 0.0
        for i in range(2):
            env.dynamic_total += env.materials[i].value(2.0)
        sim.suspend()

    graph_nodes = {node.key for node in model.process_dag().nodes}
    assert {
        "function:materials__value__variant_0",
        "function:materials__value__variant_1",
        "function:materials__policy__calculate__variant_0",
        "function:materials__policy__calculate__variant_1",
    } <= graph_nodes

    exp = model.experiment(replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp["materials__result"][0].tolist() == [6.0, 9.0]
    assert exp["dynamic_total"][0] == 10.0


def test_heterogeneous_component_process_and_collect_overrides_run():
    class Worker(sim.Component):
        count: sim.State
        result: sim.Output

        @sim.process
        def run(self, env):
            self.count += 1
            sim.suspend()

        @sim.collect
        def report(self, env):
            self.result = self.count

    class FastWorker(Worker):
        @sim.process
        def run(self, env):
            self.count += 5
            sim.suspend()

        @sim.collect
        def report(self, env):
            self.result = self.count * 2

    class System(sim.Model):
        workers: list[Worker] = [Worker(), FastWorker(), Worker()]

    model = System()
    assert [process.name for process in model._processes] == [
        "workers__variant_0__run",
        "workers__variant_1__run",
    ]
    exp = model.experiment(replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp["workers__result"][0].tolist() == [1.0, 10.0, 1.0]


def test_long_polymorphic_process_names_fit_native_buffer():
    class Worker(sim.Component):
        completed: sim.State

        @sim.process
        def perform_lengthy_operation(self, env):
            self.completed += 1
            sim.suspend()

    class FastWorker(Worker):
        @sim.process
        def perform_lengthy_operation(self, env):
            self.completed += 2
            sim.suspend()

    class System(sim.Model):
        extraordinarily_long_worker_collection: list[Worker] = [
            Worker(),
            FastWorker(),
        ]

    model = System()
    assert all(
        len(process.name.encode("utf-8")) > 31
        for process in model._processes
    )
    exp = model.experiment(replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp.trials[
        "extraordinarily_long_worker_collection__completed"
    ][0].tolist() == [1, 2]


def test_subtype_only_entity_and_processes_fields_are_packed():
    class Policy(sim.Component):
        capacity: sim.Param

        def __init__(self, capacity):
            self.capacity = capacity

    class ManagedPolicy(Policy):
        queue: sim.Queue = sim.capacity("capacity")
        workers: sim.Processes
        result: sim.Output

        @sim.process(copies=2)
        def workers(self, env, idx):
            self.queue.put(1)
            sim.suspend()

        @sim.collect
        def report(self, env):
            self.result = self.queue.level()

    class System(sim.Model):
        policies: list[Policy] = [Policy(1), ManagedPolicy(2)]

    model = System()
    assert model.dtype["policies__queue"].shape == ()
    assert model.dtype["policies__workers"].shape == (2,)
    assert model.dtype["policies__result"].shape == ()
    exp = model.experiment(replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp["policies__result"][0] == 2.0
    assert all(exp.trials["policies__workers"][0] != 0)


def test_polymorphic_subtype_pqueues_use_group_and_packed_offsets():
    class Policy(sim.Component):
        pass

    class PriorityPolicy(Policy):
        lane_count: sim.Const[int]
        lanes: sim.PQueues = sim.count("lane_count")
        result: sim.State

        def __init__(self, lane_count):
            self.lane_count = lane_count

        @sim.process
        def run(self, env):
            lane = self.lanes[self.lane_count - 1]
            lane.put(4, 0)
            self.result += lane.take()
            sim.suspend()

    class System(sim.Model):
        policies: list[Policy] = [
            PriorityPolicy(1),
            Policy(),
            PriorityPolicy(2),
        ]

    model = System()
    assert model.pqueues == {"policies__lanes": 3}
    assert [process.name for process in model._processes] == [
        "policies__variant_0__run",
    ]
    exp = model.experiment(replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp.trials["policies__result"][0].tolist() == [4, 4]


def test_polymorphic_subtype_spawnables_use_packed_descriptors():
    class Policy(sim.Component):
        pass

    class Flow(Policy):
        amount: sim.Const[int]
        count: sim.State
        worker: sim.Spawnable

        def __init__(self, amount):
            self.amount = amount

        @sim.process
        def worker(self, env):
            self.count += self.amount

        @sim.process
        def launch(self, env):
            handle = sim.spawn(self.worker, env)
            sim.wait_process(handle)
            sim.despawn(handle)
            sim.suspend()

    class System(sim.Model):
        policies: list[Policy] = [Policy(), Flow(2), Flow(5)]

    model = System()
    worker_bindings = [
        (process.name, process.spawn_index)
        for process in model._processes
        if process.spawnable
    ]
    assert worker_bindings == [
        ("policies__1__worker", 0),
        ("policies__2__worker", 1),
    ]
    exp = model.experiment(replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp.trials["policies__count"][0].tolist() == [2, 5]


def test_polymorphic_ref_and_refs_dispatch_concrete_functions():
    class Policy(sim.Component):
        @sim.function
        def calculate(self, value: float) -> float:
            return value

    class DoublePolicy(Policy):
        @sim.function
        def calculate(self, value: float) -> float:
            return value * 2.0

    class TriplePolicy(Policy):
        @sim.function
        def calculate(self, value: float) -> float:
            return value * 3.0

    class Follower(sim.Component):
        target: sim.Ref[Policy]
        result: sim.FloatState

        def __init__(self, target):
            self.target = target

        @sim.process
        def run(self, env):
            self.result = self.target.calculate(2.0)
            sim.suspend()

    class Router(sim.Component):
        routes: sim.Refs[Policy]
        result: sim.FloatState

        def __init__(self, routes):
            self.routes = routes

        @sim.process
        def run(self, env):
            for i in range(2):
                self.result += self.routes[i].calculate(2.0)
            sim.suspend()

    double = DoublePolicy()
    triple = TriplePolicy()

    class System(sim.Model):
        policies: list[Policy] = [double, triple]
        followers: list[Follower] = [
            Follower(double),
            Follower(triple),
        ]
        router: Router = Router((double, triple))

    model = System()
    exp = model.experiment(replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp.trials["followers__result"][0].tolist() == [4.0, 6.0]
    assert exp.trials["router__result"][0] == 10.0


def test_polymorphic_component_errors_are_clear():
    class Policy(sim.Component):
        @sim.function
        def calculate(self, value: float) -> float:
            return value

    class WithField(Policy):
        amount: sim.Param = 2.0

    class WithoutField(Policy):
        pass

    class DynamicFields(sim.Model):
        policies: list[Policy] = [WithField(), WithoutField()]
        result: sim.Output

    model = DynamicFields()
    with pytest.raises(ValueError, match="not declared by every concrete"):
        @model.process
        def bad_field(env: DynamicFields):
            i = env.result
            env.result = env.policies[i].amount

    class IntPolicy(Policy):
        @sim.function
        def calculate(self, value: int) -> float:
            return float(value)

    class BadFunctions(sim.Model):
        policies: list[Policy] = [Policy(), IntPolicy()]
        result: sim.Output

    bad_functions = BadFunctions()
    with pytest.raises(TypeError, match="incompatible signatures"):
        @bad_functions.process
        def bad_call(env: BadFunctions):
            i = 0
            env.result = env.policies[i].calculate(1.0)

    class ParamPolicy(Policy):
        setting: sim.Param

    class StatePolicy(Policy):
        setting: sim.State

    class BadFields(sim.Model):
        policies: list[Policy] = [ParamPolicy(), StatePolicy()]

    with pytest.raises(TypeError, match="incompatible field 'setting'"):
        BadFields()


def test_component_function_reads_params_and_returns_value():
    class Policy(sim.Component):
        threshold: sim.Param
        target: sim.Param

        @sim.function
        def decide(self, level: float) -> float:
            return self.target - level if level < self.threshold else 0.0

    class System(sim.Model):
        policy: Policy = Policy()
        level: sim.Param
        order: sim.Output

    model = System()

    @model.process
    def run(env: System):
        env.order = env.policy.decide(env.level)

    source = model._processes[0].fn.__cimba_source__
    assert "_CIMBA_FUNCTION_Policy_decide_" in source
    assert "env.policy__threshold" in source
    assert "env.policy__target" in source
    (spec,) = model._component_functions.values()
    assert spec.helper.nopython_signatures
    assert "__cimba_dep_0" in spec.helper.__cimba_source__
    graph = model.process_dag()
    assert {node.key for node in graph.nodes} >= {
        "process:run",
        "function:policy__decide",
        "param:policy__threshold",
        "param:policy__target",
    }
    assert {
        (edge.source, edge.target, edge.label) for edge in graph.edges
    } >= {
        ("process:run", "function:policy__decide", "call"),
        ("param:policy__threshold", "function:policy__decide", "read"),
        ("param:policy__target", "function:policy__decide", "read"),
    }
    (block,) = graph.blocks
    assert "function:policy__decide" in block.members
    assert "param:policy__threshold" in block.members
    assert 'n_function_policy__decide[["policy__decide"]]' \
        in graph.to_mermaid()
    assert ('"function:policy__decide" '
            '[label="policy__decide", shape=component];') in graph.to_dot()

    exp = model.experiment(
        policy__threshold=30.0,
        policy__target=80.0,
        level=20.0,
        replications=1,
        duration=1.0,
    )
    assert exp.run() == 0
    assert exp["order"][0] == 60.0


def test_component_function_can_be_called_from_component_process():
    class Policy(sim.Component):
        scale: sim.Param
        result: sim.Output

        @sim.function
        def apply(self, value: float) -> float:
            return self.scale * value

        @sim.process
        def run(self, env):
            self.result = self.apply(4.0)

    class System(sim.Model):
        policy: Policy = Policy()

    model = System()
    edges = {
        (edge.source, edge.target, edge.label)
        for edge in model.process_dag().edges
    }
    assert (
        "process:policy__run",
        "function:policy__apply",
        "call",
    ) in edges
    exp = model.experiment(
        policy__scale=2.5, replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp["policy__result"][0] == 10.0


def test_component_function_collection_dynamic_index():
    class Policy(sim.Component):
        offset: sim.Param

        @sim.function
        def apply(self, value: float) -> float:
            return self.offset + value

    class System(sim.Model):
        policies: list[Policy] = [Policy(), Policy()]
        choice: sim.State
        result: sim.Output

    model = System()

    @model.process
    def run(env: System):
        env.choice = 1
        env.result = env.policies[env.choice].apply(3.0)

    exp = model.experiment(
        policies__offset=[[2.0, 7.0]],
        replications=1,
        duration=1.0,
    )
    assert exp.run() == 0
    assert exp["result"][0] == 10.0


def test_component_function_reads_state_float_state_output_and_const():
    class Policy(sim.Component):
        offset: sim.Const[int]
        count: sim.State
        level: sim.FloatState
        prior: sim.Output
        result: sim.Output

        def __init__(self):
            self.offset = 4

        @sim.function
        def value(self, add: int) -> float:
            return self.count + self.level + self.prior + self.offset + add

        @sim.process
        def run(self, env):
            self.count = 2
            self.level = 1.5
            self.prior = 3.0
            self.result = self.value(5)

    class System(sim.Model):
        policy: Policy = Policy()

    exp = System().experiment(replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp["policy__result"][0] == 15.5


def test_component_function_calls_component_function():
    class Policy(sim.Component):
        scale: sim.Param
        bias: sim.Param

        @sim.function
        def scaled(self, value: float) -> float:
            return self.scale * value

        @sim.function
        def decide(self, value: float) -> float:
            return self.scaled(value) + self.bias

    class System(sim.Model):
        policy: Policy = Policy()
        result: sim.Output

    model = System()

    @model.process
    def run(env: System):
        env.result = env.policy.decide(4.0)

    exp = model.experiment(
        policy__scale=2.0,
        policy__bias=3.0,
        replications=1,
        duration=1.0,
    )
    assert exp.run() == 0
    assert exp["result"][0] == 11.0
    edges = {
        (edge.source, edge.target, edge.label)
        for edge in model.process_dag().edges
    }
    assert ("process:run", "function:policy__decide", "call") in edges
    assert (
        "function:policy__decide",
        "function:policy__scaled",
        "call",
    ) in edges


def test_component_function_nested_component_ref_and_collect():
    class Rule(sim.Component):
        amount: sim.Param

        @sim.function
        def value(self, scale: float) -> float:
            return self.amount * scale

    class Wrapper(sim.Component):
        rule: Rule = Rule()

        @sim.function
        def decide(self, scale: float) -> float:
            return self.rule.value(scale)

    class Reporter(sim.Component):
        rule: sim.Ref[Rule]
        result: sim.Output

        def __init__(self, rule):
            self.rule = rule

        @sim.collect
        def report(self, env):
            self.result = self.rule.value(3.0)

    rule = Rule()

    class System(sim.Model):
        direct: Rule = rule
        wrapper: Wrapper = Wrapper()
        reporter: Reporter = Reporter(rule)
        result: sim.Output
        collected: sim.Output

    model = System()

    @model.process
    def run(env: System):
        env.result = env.wrapper.decide(2.0)

    @model.collect
    def collect(env: System):
        env.collected = env.direct.value(4.0)

    exp = model.experiment(
        direct__amount=4.0,
        wrapper__rule__amount=5.0,
        replications=1,
        duration=1.0,
    )
    assert exp.run() == 0
    assert exp["result"][0] == 10.0
    assert exp["reporter__result"][0] == 12.0
    assert exp["collected"][0] == 16.0


def test_component_function_classes_and_cache_are_independent():
    class Add(sim.Component):
        value: sim.Param

        @sim.function
        def decide(self, x: float) -> float:
            return x + self.value

    class Multiply(sim.Component):
        value: sim.Param

        @sim.function
        def decide(self, x: float) -> float:
            return x * self.value

    class System(sim.Model):
        add: Add = Add()
        other_add: Add = Add()
        multiply: Multiply = Multiply()
        added: sim.Output
        multiplied: sim.Output

    model = System()
    add_specs = [
        spec for spec in model._component_functions.values()
        if spec.decl.cls is Add
    ]
    assert len(add_specs) == 2
    assert add_specs[0].helper is add_specs[1].helper

    @model.process
    def run(env: System):
        env.added = env.add.decide(3.0)
        env.multiplied = env.multiply.decide(3.0)

    exp = model.experiment(
        add__value=4.0,
        other_add__value=9.0,
        multiply__value=4.0,
        replications=1,
        duration=1.0,
    )
    assert exp.run() == 0
    assert exp["added"][0] == 7.0
    assert exp["multiplied"][0] == 12.0


def test_component_function_rejects_invalid_signatures_and_calls():
    class MissingAnnotation(sim.Component):
        @sim.function
        def bad(self, value) -> float:
            return value

    class MissingReturn(sim.Component):
        @sim.function
        def bad(self, value: float):
            return value

    class Unsupported(sim.Component):
        @sim.function
        def bad(self, value: list) -> float:
            return 0.0

    for component, match in (
        (MissingAnnotation(), "argument 'value' needs"),
        (MissingReturn(), "return type annotation"),
        (Unsupported(), "unsupported type annotation"),
    ):
        class System(sim.Model):
            item: type(component) = component

        with pytest.raises(TypeError, match=match):
            System()

    class Valid(sim.Component):
        @sim.function
        def apply(self, value: float) -> float:
            return value

    class ValidSystem(sim.Model):
        item: Valid = Valid()

    model = ValidSystem()
    with pytest.raises(ValueError, match="must use positional arguments"):
        @model.process
        def run(env: ValidSystem):
            env.item.apply(value=1.0)

    class BadBody(sim.Component):
        @sim.function
        def bad(self, value: float) -> float:
            return "not a scalar"

    class BadBodySystem(sim.Model):
        item: BadBody = BadBody()

    with pytest.raises(TypeError, match="failed Numba nopython compilation"):
        BadBodySystem()


def test_component_function_rejects_mutation_scheduling_and_entity_calls():
    class Mutating(sim.Component):
        value: sim.State

        @sim.function
        def bad(self, amount: int) -> int:
            self.value += amount
            return self.value

    class Scheduling(sim.Component):
        @sim.function
        def bad(self, delay: float) -> float:
            sim.hold(delay)
            return delay

    class EntityCall(sim.Component):
        queue: sim.Queue

        @sim.function
        def bad(self, amount: int) -> int:
            return self.queue.level() + amount

    for component, match in (
        (Mutating(), "cannot mutate component field"),
        (Scheduling(), "cannot call scheduling/process operation"),
        (EntityCall(), "cannot call component field operation"),
    ):
        class System(sim.Model):
            item: type(component) = component

        with pytest.raises(ValueError, match=match):
            System()


def test_component_function_rejects_recursive_calls_and_marker_conflicts():
    class Recursive(sim.Component):
        @sim.function
        def first(self, value: float) -> float:
            return self.second(value)

        @sim.function
        def second(self, value: float) -> float:
            return self.first(value)

    class System(sim.Model):
        item: Recursive = Recursive()

    with pytest.raises(ValueError, match="recursive component function call"):
        System()

    def process_then_function(self, env):
        pass

    with pytest.raises(ValueError, match="component function"):
        sim.function(sim.process(process_then_function))

    def function_then_process(self, env):
        pass

    with pytest.raises(ValueError, match="component function"):
        sim.process(sim.function(function_then_process))

    def collect_then_function(self, env):
        pass

    with pytest.raises(ValueError, match="component function"):
        sim.function(sim.collect(collect_then_function))

    def function_then_collect(self, env):
        pass

    with pytest.raises(ValueError, match="component function"):
        sim.collect(sim.function(function_then_collect))

    class Callable(sim.Component):
        @sim.function
        def value(self, amount: float) -> float:
            return amount

    class CallableSystem(sim.Model):
        item: Callable = Callable()

    model = CallableSystem()
    with pytest.raises(ValueError, match="cannot be registered"):
        model.process(model.item.value)


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
            self.orders.put(self.B)
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


def test_component_param_defaults_flatten_per_instance_and_can_be_overridden():
    class Policy(sim.Component):
        threshold: sim.Param = 10.0
        multiplier: sim.Param

        def __init__(self, multiplier: float):
            self.multiplier = multiplier

    class Network(sim.Model):
        policies: list[Policy] = [Policy(2.0), Policy(3.0)]
        result: sim.Output

    model = Network()
    assert model.param_defaults == {
        "policies__threshold": (10.0, 10.0),
        "policies__multiplier": (2.0, 3.0),
    }

    @model.process
    def run(env: Network):
        env.result = (
            env.policies[0].threshold * env.policies[0].multiplier
            + env.policies[1].threshold * env.policies[1].multiplier
        )

    defaulted = model.experiment(replications=1, duration=1.0)
    assert defaulted.run() == 0
    assert defaulted["result"][0] == 50.0
    assert defaulted.trials["policies__threshold"][0].tolist() == [10.0, 10.0]

    overridden = model.experiment(
        policies__threshold=[5.0, 7.0],
        policies__multiplier=[4.0, 6.0],
        replications=1,
        duration=1.0,
    )
    assert overridden.run() == 0
    assert overridden["result"][0] == 62.0


def test_component_param_defaults_must_cover_every_collection_item():
    class Policy(sim.Component):
        threshold: sim.Param

        def __init__(self, threshold=None):
            if threshold is not None:
                self.threshold = threshold

    class Network(sim.Model):
        policies: list[Policy] = [Policy(2.0), Policy()]

    with pytest.raises(
            ValueError, match="must have a default on every instance or none"):
        Network()


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
    # Block members list the processes first, then the component's
    # entities in declaration order.
    assert graph.blocks == (
        sim.ProcessDAGBlock(
            "retailer",
            (
                "process:retailer__reorder",
                "state:retailer__on_hand",
                "queue:retailer__orders",
            ),
        ),
    )
    assert 'subgraph n_component_retailer["retailer"]' \
        in graph.to_mermaid()
    dot = graph.to_dot()
    assert "subgraph cluster_component_retailer {" in dot
    assert 'label="retailer";' in dot


def test_model_collect_can_use_component_field_namespace():
    class Station(sim.Component):
        queue: sim.Queue

    class Network(sim.Model):
        avg: sim.Output
        station: Station = Station()

    model = Network()

    @model.process
    def feeder(env: Network):
        env.station.queue.put(2)
        sim.suspend()

    @model.collect
    def collect_stats(env: Network):
        env.avg = env.station.queue.mean_level()

    assert "env.station__queue" in model._processes[0].fn.__cimba_source__
    assert "env.station__queue" in model._collect.__cimba_source__
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=13)
    assert exp.run() == 0
    assert exp["avg"][0] == 2.0


def test_component_collect_assigns_own_outputs():
    class Station(sim.Component):
        avg: sim.Output
        queue: sim.Queue

        @sim.process
        def feeder(self, env):
            self.queue.put(2)
            sim.suspend()

        @sim.collect
        def station_stats(self, env):
            self.avg = self.queue.mean_level()

    class Network(sim.Model):
        station: Station = Station()

    model = Network()
    ((lowered, _instances),) = model._component_collects
    assert "env.station__queue" in lowered.__cimba_source__
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=21)
    assert exp.run() == 0
    assert exp["station__avg"][0] == 2.0


def test_component_collect_can_use_dataset_methods():
    class Sampler(sim.Component):
        avg: sim.Output
        samples: sim.Dataset

        @sim.process
        def feeder(self, env):
            self.samples.add(1.0)
            self.samples.add(3.0)

        @sim.collect
        def sampler_stats(self, env):
            self.avg = self.samples.mean()

    class Network(sim.Model):
        sampler: Sampler = Sampler()

    model = Network()
    ((lowered, _instances),) = model._component_collects
    assert (
        "_cimba_dataset_mean(env.sampler__samples)"
        in lowered.__cimba_source__
    )
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=24)
    assert exp.run() == 0
    assert exp["sampler__avg"][0] == 2.0


def test_component_methods_can_use_model_dataset_methods():
    class Sampler(sim.Component):
        avg: sim.Output

        @sim.process
        def feeder(self, env):
            env.samples.add(2.0)
            env.samples.add(4.0)

        @sim.collect
        def sampler_stats(self, env):
            self.avg = env.samples.mean()

    class Network(sim.Model):
        samples: sim.Dataset
        sampler: Sampler = Sampler()

    model = Network()
    assert (
        "_cimba_dataset_add(env.samples"
        in model._processes[0].fn.__cimba_source__
    )
    ((lowered, _instances),) = model._component_collects
    assert "_cimba_dataset_mean(env.samples)" in lowered.__cimba_source__
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=25)
    assert exp.run() == 0
    assert exp["sampler__avg"][0] == 3.0


def test_component_collection_collects_run_per_instance_before_model():
    class Desk(sim.Component):
        served: sim.Output
        waiting: sim.Queue

        def __init__(self, amount: int):
            self.amount = amount

        @sim.process
        def feeder(self, env):
            self.waiting.put(self.amount)
            sim.suspend()

        @sim.collect
        def desk_stats(self, env):
            self.served = self.waiting.mean_level()

    class Clinic(sim.Model):
        total: sim.Output
        desks: list[Desk] = [Desk(2), Desk(5)]

    model = Clinic()

    @model.collect
    def clinic_stats(env: Clinic):
        env.total = env.desks[0].served + env.desks[1].served

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=22)
    assert exp.run() == 0
    assert exp["desks__served"][0].tolist() == [2.0, 5.0]
    assert exp["total"][0] == 7.0


def test_nested_component_collect_runs():
    class Inner(sim.Component):
        done: sim.Output
        count: sim.State

        @sim.collect
        def inner_stats(self, env):
            self.done = self.count + 1.0

    class Outer(sim.Component):
        inner: Inner = Inner()

        @sim.process
        def actor(self, env):
            self.inner.count = 4
            sim.suspend()

    class Network(sim.Model):
        outer: Outer = Outer()

    model = Network()
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=23)
    assert exp.run() == 0
    assert exp["outer__inner__done"][0] == 5.0


def test_component_collect_declaration_errors_are_rejected():
    with pytest.raises(ValueError, match="cannot be both"):
        class CollectOverProcess(sim.Component):
            @sim.collect
            @sim.process
            def stats(self, env):
                pass

    with pytest.raises(ValueError, match="cannot be both"):
        class ProcessOverCollect(sim.Component):
            @sim.process
            @sim.collect
            def stats(self, env):
                pass

    class BadSignature(sim.Component):
        total: sim.Output

        @sim.collect
        def stats(self, env, idx):
            self.total = idx

    class BadSignatureModel(sim.Model):
        bad: BadSignature = BadSignature()

    with pytest.raises(ValueError, match=r"must take \(self, env\)"):
        BadSignatureModel()

    class CallsHelper(sim.Component):
        total: sim.Output

        def helper(self):
            return 1.0

        @sim.collect
        def stats(self, env):
            self.total = self.helper()

    class CallsHelperModel(sim.Model):
        station: CallsHelper = CallsHelper()

    with pytest.raises(ValueError,
                       match="collect cannot call self.helper"):
        CallsHelperModel()


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
        env.gate.signal()
        sim.suspend()

    @model.process
    def waiter(env: Network):
        env.gate.wait_for(env.ready)
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
        env.bump.schedule(1.0)
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
        env.station.queue.put(1)
        sim.suspend()

    graph = model.process_dag()
    nodes = {node.key for node in graph.nodes}
    edges = {(edge.source, edge.target, edge.label) for edge in graph.edges}
    assert "queue:station__queue" in nodes
    assert "state:station__count" in nodes
    assert ("process:feeder", "queue:station__queue", "put") in edges
    assert ("process:feeder", "state:station__count", "write") in edges
    assert ("state:station__count", "process:feeder", "read") in edges
    assert graph.blocks == (
        sim.ProcessDAGBlock(
            "station",
            ("queue:station__queue", "state:station__count"),
        ),
    )


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


def test_component_collection_declarations_flatten_to_shaped_fields():
    class Attraction(sim.Component):
        active: sim.State
        score: sim.Output
        queue: sim.Queue
        lanes: sim.PQueues = sim.count("queue_count")

        def __init__(self, queue_count: int, bias: int):
            self.queue_count = queue_count
            self.bias = bias

    class Park(sim.Model):
        attractions: list[Attraction] = [
            Attraction(queue_count=1, bias=10),
            Attraction(queue_count=3, bias=20),
        ]

    model = Park()
    assert model.state == ["attractions__active"]
    assert model.outputs == ["attractions__score"]
    assert model.queues == {"attractions__queue": None}
    assert model.pqueues == {"attractions__lanes": 4}
    assert model.dtype["attractions__active"].shape == (2,)
    assert model.dtype["attractions__score"].shape == (2,)
    assert model.dtype["attractions__queue"].shape == (2,)
    assert model.dtype["attractions__lanes"].shape == (4,)
    (decl,) = model._component_collection_decls
    assert decl.constants["queue_count"] == (1, 3)
    assert decl.constants["bias"] == (10, 20)
    assert decl.pqueue_counts["lanes"] == (1, 3)
    assert decl.pqueue_offsets["lanes"] == (0, 1)


def test_component_collection_shorthand_annotation_is_accepted():
    class Item(sim.Component):
        count: sim.State

    class Network(sim.Model):
        items: [Item] = [Item(), Item()]

    model = Network()
    assert model.dtype["items__count"].shape == (2,)


def test_model_process_can_index_component_collection_fields_and_constants():
    class Attraction(sim.Component):
        visits: sim.State
        lanes: sim.PQueues = sim.count("queue_count")

        def __init__(self, queue_count: int, bias: int):
            self.queue_count = queue_count
            self.bias = bias

    class Park(sim.Model):
        total: sim.Output
        attractions: list[Attraction] = [
            Attraction(queue_count=1, bias=10),
            Attraction(queue_count=2, bias=20),
        ]

    model = Park()

    @model.process
    def visitor(env: Park):
        at = 1
        qi = 1
        q = env.attractions[at].lanes[qi]
        q.put(env.attractions[at].bias, 0)
        env.attractions[at].visits += q.take()
        env.total = env.attractions[at].visits \
            + env.attractions[at].queue_count
        sim.suspend()

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=21)
    assert exp.run() == 0
    assert exp["total"][0] == 22.0
    assert exp.trials["attractions__visits"][0].tolist() == [0, 20]


def test_component_collection_outputs_run_and_count_failures_by_trial():
    class Item(sim.Component):
        score: sim.Output

    class Network(sim.Model):
        items: list[Item] = [Item(), Item()]

    model = Network()

    @model.process
    def actor(env: Network):
        env.items[0].score = 1.0
        env.items[1].score = 2.0
        sim.suspend()

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=23)
    assert exp.run() == 0
    assert exp["items__score"][0].tolist() == [1.0, 2.0]


def test_component_collection_can_own_params_and_sweep_vectors():
    class Item(sim.Component):
        rate: sim.Param
        score: sim.Output

        def __init__(self, bias: int):
            self.bias = bias

        @sim.process
        def record(self, env):
            self.score = self.rate + self.bias

    class Network(sim.Model):
        second_rate: sim.Output
        items: list[Item] = [Item(10), Item(20)]

    model = Network()

    @model.process
    def inspect_second(env: Network):
        env.second_rate = env.items[1].rate

    rates = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    seeds = model.trial_seeds(seed=35, items__rate=rates, replications=2)

    assert model.params == ["items__rate"]
    assert model.dtype["items__rate"].shape == (2,)
    assert len(seeds) == 4
    with pytest.raises(ValueError, match="parameter 'items__rate'"):
        model.trial_seeds(seed=35, items__rate=[1.0, 2.0, 3.0])

    exp = model.experiment(
        items__rate=rates,
        replications=2,
        duration=1.0,
        warmup=0.0,
        seed=35,
    )
    assert exp.run() == 0
    assert np.array_equal(exp["seed"], seeds)
    assert exp.trials["items__rate"].tolist() == [
        [1.0, 2.0],
        [1.0, 2.0],
        [3.0, 4.0],
        [3.0, 4.0],
    ]
    assert exp["items__score"].tolist() == [
        [11.0, 22.0],
        [11.0, 22.0],
        [13.0, 24.0],
        [13.0, 24.0],
    ]
    assert exp["second_rate"].tolist() == [2.0, 2.0, 4.0, 4.0]


def test_component_collection_params_can_size_entity_capacities():
    class Item(sim.Component):
        cap: sim.Param
        queue: sim.Queue = sim.capacity("cap")
        open_space: sim.Output

        @sim.process
        def inspect(self, env):
            self.open_space = self.queue.space()

    class Network(sim.Model):
        items: list[Item] = [Item(), Item()]

    exp = Network().experiment(
        items__cap=[2.0, 5.0],
        replications=1,
        duration=1.0,
        warmup=0.0,
        seed=36,
    )
    assert exp.run() == 0
    assert exp["items__open_space"][0].tolist() == [2.0, 5.0]


def test_component_collection_can_own_traces():
    class Source(sim.Component):
        demand: sim.Trace
        total: sim.Output

        @sim.process
        def consume(self, env):
            values = sim.Trace(self.demand)
            total = 0.0
            for value in values:
                total += value
            self.total = total

    class Network(sim.Model):
        second_first: sim.Output
        sources: list[Source] = [Source(), Source()]

    model = Network()

    @model.process
    def inspect_second(env: Network):
        values = sim.Trace(env.sources[1].demand)
        env.second_first = values[0]

    assert model.traces == ["sources__demand"]
    assert model.dtype["sources__demand"].shape == (2, 2)

    exp = model.experiment(
        sources__demand=[
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0, 5.0], dtype=np.float64),
        ],
        replications=1,
        duration=1.0,
        warmup=0.0,
        seed=36,
    )
    assert exp.run() == 0
    assert exp["sources__total"][0].tolist() == [3.0, 12.0]
    assert exp["second_first"][0] == 3.0


def test_component_collection_trace_callable_returns_component_rows():
    class Source(sim.Component):
        demand: sim.Trace
        total: sim.Output

        @sim.process
        def consume(self, env):
            values = sim.Trace(self.demand)
            total = 0.0
            for value in values:
                total += value
            self.total = total

    class Network(sim.Model):
        sources: list[Source] = [Source(), Source()]

    def generator(rng, trial):
        base = float(trial + 1)
        return np.array([
            [base, base + 1.0],
            [10.0 * base, 20.0 * base],
        ])

    exp = Network().experiment(
        sources__demand=generator,
        replications=2,
        duration=1.0,
        warmup=0.0,
        seed=37,
    )
    assert exp.run() == 0
    assert exp["sources__total"].tolist() == [[3.0, 30.0], [5.0, 60.0]]


def test_component_collection_processes_run_per_item_with_symbolic_copies():
    class Attraction(sim.Component):
        queue_count: sim.Const[int]
        server_count: sim.Const[int]
        done: sim.State
        lanes: sim.PQueues = sim.count("queue_count")

        def __init__(self, queue_count: int, server_count: int, base: int):
            self.queue_count = queue_count
            self.server_count = server_count
            self.base = base

        @sim.process(copies="server_count", priority=6)
        def server(self, env, idx):
            q = self.lanes[idx % self.queue_count]
            q.put(self.base + idx, 0)
            self.done += q.take()
            sim.suspend()

    class Park(sim.Model):
        total: sim.Output
        attractions: list[Attraction] = [
            Attraction(queue_count=1, server_count=1, base=10),
            Attraction(queue_count=2, server_count=2, base=20),
        ]

    model = Park()
    # The collection's copies are compiled into one indexed process; the
    # per-item copy counts sum, and the copy index is mapped back to the
    # owning item at runtime through a generated table.
    assert [(p.name, p.copies, p.priority, p.indexed)
            for p in model._processes] == [
        ("attractions__server", 3, 6, True),
    ]
    assert "_CIMBA_PROCINST_attractions__server" in \
        model._processes[0].fn.__cimba_source__

    @model.collect
    def collect_stats(env: Park):
        env.total = env.attractions[0].done + env.attractions[1].done

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=22)
    assert exp.run() == 0
    assert exp["total"][0] == 51.0
    assert exp.trials["attractions__done"][0].tolist() == [10, 41]


def test_component_collection_process_dag_uses_lowered_source():
    class Attraction(sim.Component):
        visits: sim.State
        lanes: sim.PQueues = sim.count("queue_count")

        def __init__(self, queue_count: int):
            self.queue_count = queue_count

        @sim.process
        def server(self, env):
            self.lanes[0].put(3, 0)
            self.visits += self.lanes[0].take()
            sim.suspend()

    class Park(sim.Model):
        attractions: list[Attraction] = [
            Attraction(queue_count=1),
            Attraction(queue_count=2),
        ]

    model = Park()

    @model.process
    def visitor(env: Park):
        at = 1
        env.attractions[at].lanes[0].put(7, 0)
        env.attractions[at].visits += 1
        sim.suspend()

    graph = model.process_dag()
    nodes = {node.key for node in graph.nodes}
    edges = {(edge.source, edge.target, edge.label) for edge in graph.edges}
    assert "pqueues:attractions__lanes" in nodes
    assert "state:attractions__visits" in nodes
    assert ("process:visitor", "pqueues:attractions__lanes", "pq_put") \
        in edges
    assert ("process:visitor", "state:attractions__visits", "write") \
        in edges
    (block,) = graph.blocks
    assert block.name == "attractions"
    assert block.kind == "component_collection"
    assert set(block.members) >= {
        "process:attractions__server",
        "pqueues:attractions__lanes",
        "state:attractions__visits",
    }
    assert 'subgraph n_component_collection_attractions["attractions"]' \
        in graph.to_mermaid()
    assert "subgraph cluster_component_collection_attractions {" \
        in graph.to_dot()


def test_nested_component_fields_processes_and_model_refs_run():
    class Counter(sim.Component):
        count: sim.State

        @sim.process
        def tick(self, env):
            self.count += 4
            sim.suspend()

    class Wrapper(sim.Component):
        counter: Counter = Counter()

        @sim.process
        def bump(self, env):
            self.counter.count += 2
            sim.suspend()

    class Network(sim.Model):
        total: sim.Output
        wrapper: Wrapper = Wrapper()

    model = Network()
    assert model.state == ["wrapper__counter__count"]
    assert [p.name for p in model._processes] == [
        "wrapper__bump",
        "wrapper__counter__tick",
    ]
    assert isinstance(model.wrapper.counter, Counter)

    @model.collect
    def collect_stats(env: Network):
        env.total = env.wrapper.counter.count

    assert "env.wrapper__counter__count" in \
        model._processes[0].fn.__cimba_source__
    assert "env.wrapper__counter__count" in model._collect.__cimba_source__
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=24)
    assert exp.run() == 0
    assert exp["total"][0] == 6.0
    assert exp.trials["wrapper__counter__count"][0] == 6


def test_component_collection_can_own_nested_component_processes():
    class Servers(sim.Component):
        done: sim.State

        def __init__(self, server_count: int, base: int):
            self.server_count = server_count
            self.base = base

        @sim.process(copies="server_count", priority=3)
        def server(self, env, idx):
            self.done += self.base + idx
            sim.suspend()

    class Attraction(sim.Component):
        def __init__(self, server_count: int, base: int):
            self.servers = Servers(server_count, base)

        servers: Servers

    class Park(sim.Model):
        total: sim.Output
        attractions: list[Attraction] = [
            Attraction(server_count=1, base=10),
            Attraction(server_count=2, base=20),
        ]

    model = Park()
    assert [(p.name, p.copies, p.priority, p.indexed)
            for p in model._processes] == [
        ("attractions__servers__server", 3, 3, True),
    ]
    assert model.dtype["attractions__servers__done"].shape == (2,)
    assert isinstance(model.attractions[0].servers, Servers)

    @model.collect
    def collect_stats(env: Park):
        env.total = env.attractions[0].servers.done \
            + env.attractions[1].servers.done

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=25)
    assert exp.run() == 0
    assert exp["total"][0] == 51.0
    assert exp.trials["attractions__servers__done"][0].tolist() == [10, 41]


def test_nested_component_collections_linearize_indexes():
    class Gate(sim.Component):
        count: sim.State

    class Zone(sim.Component):
        gates: list[Gate]

        def __init__(self, gates: int):
            self.gates = [Gate() for _ in range(gates)]

    class Campus(sim.Model):
        total: sim.Output
        zones: list[Zone] = [Zone(1), Zone(2)]

    model = Campus()
    assert model.dtype["zones__gates__count"].shape == (3,)
    (zones,) = model._component_collection_decls
    (gates,) = zones.children
    assert gates.parent_offsets == (0, 1)
    assert gates.parent_lengths == (1, 2)

    @model.process
    def actor(env: Campus):
        env.zones[1].gates[1].count = 7
        env.total = env.zones[1].gates[1].count
        sim.suspend()

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=26)
    assert exp.run() == 0
    assert exp["total"][0] == 7.0
    assert exp.trials["zones__gates__count"][0].tolist() == [0, 0, 7]


def test_nested_component_collection_pqueues_use_nested_offsets():
    class Gate(sim.Component):
        hits: sim.State
        lanes: sim.PQueues = sim.count("lane_count")

        def __init__(self, lane_count: int):
            self.lane_count = lane_count

    class Zone(sim.Component):
        gates: list[Gate]

        def __init__(self, lane_counts: list[int]):
            self.gates = [Gate(count) for count in lane_counts]

    class Campus(sim.Model):
        total: sim.Output
        zones: list[Zone] = [Zone([1]), Zone([2, 1])]

    model = Campus()
    assert model.pqueues == {"zones__gates__lanes": 4}
    (zones,) = model._component_collection_decls
    (gates,) = zones.children
    assert gates.pqueue_counts["lanes"] == (1, 2, 1)
    assert gates.pqueue_offsets["lanes"] == (0, 1, 3)

    @model.process
    def actor(env: Campus):
        q = env.zones[1].gates[0].lanes[1]
        q.put(12, 0)
        env.zones[1].gates[0].hits += q.take()
        env.total = env.zones[1].gates[0].hits
        sim.suspend()

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=27)
    assert exp.run() == 0
    assert exp["total"][0] == 12.0
    assert exp.trials["zones__gates__hits"][0].tolist() == [0, 12, 0]


def test_nested_component_owned_spawnable_runs_with_struct_view():
    class Visitor(sim.Struct):
        weight: float

    class Entrance(sim.Component):
        total: sim.FloatState
        visitor: sim.Spawnable

        @sim.process
        def visitor(self, env, vip: Visitor):
            self.total += vip.weight

    class Flow(sim.Component):
        entrance: Entrance = Entrance()

        @sim.process
        def arrivals(self, env):
            handle = sim.spawn(self.entrance.visitor, env)
            Visitor(handle).weight = 3.5
            sim.wait_process(handle)
            sim.despawn(handle)
            sim.suspend()

    class Network(sim.Model):
        total: sim.Output
        flow: Flow = Flow()

    model = Network()
    assert model._spawnable_fields == ["flow__entrance__visitor"]
    assert [
        (p.name, p.spawnable, p.spawn_field, p.spawn_index)
        for p in model._processes
    ] == [
        ("flow__arrivals", False, None, None),
        ("flow__entrance__visitor", True, "flow__entrance__visitor", None),
    ]

    @model.collect
    def collect_stats(env: Network):
        env.total = env.flow.entrance.total

    exp = model.experiment(replications=1, duration=5.0, warmup=0.0,
                           seed=28)
    assert exp.run() == 0
    assert exp["total"][0] == 3.5


def test_component_collection_owned_spawnables_get_per_item_descriptors():
    class Flow(sim.Component):
        count: sim.State
        worker: sim.Spawnable

        def __init__(self, amount: int):
            self.amount = amount

        @sim.process
        def worker(self, env):
            self.count += self.amount

        @sim.process
        def launch(self, env):
            handle = sim.spawn(self.worker, env)
            sim.wait_process(handle)
            sim.despawn(handle)
            sim.suspend()

    class Network(sim.Model):
        flows: list[Flow] = [Flow(2), Flow(5)]

    model = Network()
    assert model.dtype["flows__worker"].shape == (2,)
    worker_bindings = [
        (p.name, p.spawn_field, p.spawn_index)
        for p in model._processes
        if p.spawnable
    ]
    assert worker_bindings == [
        ("flows__0__worker", "flows__worker", 0),
        ("flows__1__worker", "flows__worker", 1),
    ]

    exp = model.experiment(replications=1, duration=5.0, warmup=0.0,
                           seed=29)
    assert exp.run() == 0
    assert exp.trials["flows__count"][0].tolist() == [2, 5]


def test_scalar_component_can_own_process_handles():
    class Workers(sim.Component):
        total: sim.State
        worker: sim.Processes

        @sim.process(copies=2)
        def worker(self, env, idx):
            sig = sim.suspend()
            self.total += (idx + 1) * sig

        @sim.process
        def supervisor(self, env):
            sim.hold(0.1)
            sim.interrupt(self.worker[1], 7, 0)
            sim.suspend()

    class Network(sim.Model):
        total: sim.Output
        workers: Workers = Workers()

    model = Network()
    assert model.dtype["workers__worker"].shape == (2,)

    @model.process
    def outside_supervisor(env: Network):
        sim.hold(0.2)
        sim.interrupt(env.workers.worker[0], 5, 0)
        sim.suspend()

    @model.collect
    def collect_stats(env: Network):
        env.total = env.workers.total

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=33)
    assert exp.run() == 0
    assert exp["total"][0] == 19.0


def test_component_collection_can_own_ragged_process_handles():
    class Team(sim.Component):
        worker_count: sim.Const[int]
        base: sim.Const[int]
        total: sim.State
        worker: sim.Processes

        def __init__(self, worker_count: int, base: int):
            self.worker_count = worker_count
            self.base = base

        @sim.process(copies="worker_count")
        def worker(self, env, idx):
            sig = sim.suspend()
            self.total += self.base + idx + sig

    class Network(sim.Model):
        total: sim.Output
        teams: list[Team] = [Team(1, 10), Team(3, 20)]

    model = Network()
    assert model.dtype["teams__worker"].shape == (4,)
    (teams,) = model._component_collection_decls
    assert teams.process_counts["worker"] == (1, 3)
    assert teams.process_offsets["worker"] == (0, 1)

    @model.process
    def supervisor(env: Network):
        sim.hold(0.1)
        sim.interrupt(env.teams[0].worker[0], 1, 0)
        sim.interrupt(env.teams[1].worker[2], 2, 0)
        sim.suspend()

    @model.collect
    def collect_stats(env: Network):
        env.total = env.teams[0].total + env.teams[1].total

    assert "env.teams__worker[0]" in model._processes[-1].fn.__cimba_source__
    assert "env.teams__worker[3]" in model._processes[-1].fn.__cimba_source__

    graph = model.process_dag()
    # Both instances' workers share one compiled process; the supervisor's
    # interrupts of specific handles map to that single graph node.
    assert sim.ProcessDAGEdge(
        "process:supervisor",
        "process:teams__worker",
        "interrupt",
    ) in graph.edges

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=34)
    assert exp.run() == 0
    assert exp["total"][0] == 35.0


def test_nested_component_collection_can_own_process_handles():
    class Gate(sim.Component):
        total: sim.State
        worker: sim.Processes

        def __init__(self, worker_count: int, base: int):
            self.worker_count = worker_count
            self.base = base

        @sim.process(copies="worker_count")
        def worker(self, env, idx):
            sig = sim.suspend()
            self.total += self.base + idx + sig

    class Zone(sim.Component):
        gates: list[Gate]

        def __init__(self, gates: list[Gate]):
            self.gates = gates

    class Campus(sim.Model):
        total: sim.Output
        zones: list[Zone] = [
            Zone([Gate(1, 10)]),
            Zone([Gate(2, 20), Gate(1, 30)]),
        ]

    model = Campus()
    assert model.dtype["zones__gates__worker"].shape == (4,)
    (zones,) = model._component_collection_decls
    (gates,) = zones.children
    assert gates.process_counts["worker"] == (1, 2, 1)
    assert gates.process_offsets["worker"] == (0, 1, 3)

    @model.process
    def supervisor(env: Campus):
        sim.hold(0.1)
        sim.interrupt(env.zones[1].gates[0].worker[1], 4, 0)
        sim.suspend()

    @model.collect
    def collect_stats(env: Campus):
        env.total = env.zones[1].gates[0].total

    assert "env.zones__gates__worker[2]" in \
        model._processes[-1].fn.__cimba_source__
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=35)
    assert exp.run() == 0
    assert exp["total"][0] == 25.0


def test_model_process_can_spawn_component_collection_field():
    class Flow(sim.Component):
        count: sim.State
        visitor: sim.Spawnable

        @sim.process
        def visitor(self, env):
            self.count += 1

    class Network(sim.Model):
        flows: list[Flow] = [Flow(), Flow()]

    model = Network()

    @model.process
    def launcher(env: Network):
        handle = sim.spawn(env.flows[1].visitor, env)
        sim.wait_process(handle)
        sim.despawn(handle)
        sim.suspend()

    assert "env.flows__visitor[1]" in model._processes[-1].fn.__cimba_source__
    exp = model.experiment(replications=1, duration=5.0, warmup=0.0,
                           seed=30)
    assert exp.run() == 0
    assert exp.trials["flows__count"][0].tolist() == [0, 1]


def test_model_process_can_spawn_nested_component_spawnable_path():
    class Entrance(sim.Component):
        count: sim.State
        visitor: sim.Spawnable

        @sim.process
        def visitor(self, env):
            self.count += 1

    class ParkArea(sim.Component):
        entrance: Entrance = Entrance()

    class Park(sim.Model):
        park: ParkArea = ParkArea()

    model = Park()

    @model.process
    def arrivals(env: Park):
        handle = sim.spawn(env.park.entrance.visitor, env)
        sim.wait_process(handle)
        sim.despawn(handle)
        sim.suspend()

    exp = model.experiment(replications=1, duration=5.0, warmup=0.0,
                           seed=31)
    assert exp.run() == 0
    assert exp.trials["park__entrance__count"][0] == 1


def test_component_process_supports_indexed_struct_injection():
    class Tag(sim.Struct):
        value: int

    class Workers(sim.Component):
        total: sim.State

        @sim.process(copies=2)
        def worker(self, env, idx, tag: Tag):
            tag.value = idx + 1
            self.total += tag.value

    class Network(sim.Model):
        workers: Workers = Workers()

    exp = Network().experiment(replications=1, duration=1.0, warmup=0.0,
                               seed=32)
    assert exp.run() == 0
    assert exp.trials["workers__total"][0] == 3


def test_component_owned_spawnable_process_dag_edges():
    class Flow(sim.Component):
        worker: sim.Spawnable

        @sim.process
        def worker(self, env):
            sim.suspend()

        @sim.process
        def launch(self, env):
            sim.spawn(self.worker, env)
            sim.suspend()

    class Network(sim.Model):
        flows: list[Flow] = [Flow(), Flow()]

    graph = Network().process_dag()
    edges = {(edge.source, edge.target, edge.label) for edge in graph.edges}
    # The launcher is compiled once for the collection and spawns whichever
    # item's worker its runtime instance selects, so it reaches both.
    assert {
        ("process:flows__launch", "process:flows__0__worker", "spawn"),
        ("process:flows__launch", "process:flows__1__worker", "spawn"),
    } <= edges


def test_component_spawnable_declaration_errors_are_rejected():
    class MissingWorker(sim.Component):
        worker: sim.Spawnable

    class MissingModel(sim.Model):
        flow: MissingWorker = MissingWorker()

    missing = MissingModel()

    @missing.process
    def idle(env: MissingModel):
        sim.suspend()

    with pytest.raises(ValueError, match="flow__worker"):
        missing.experiment()

    class CopiesWorker(sim.Component):
        worker: sim.Spawnable

        @sim.process(copies=2)
        def worker(self, env):
            sim.suspend()

    class CopiesModel(sim.Model):
        flow: CopiesWorker = CopiesWorker()

    with pytest.raises(ValueError, match="cannot take copies"):
        CopiesModel()

    class IndexedWorker(sim.Component):
        worker: sim.Spawnable

        @sim.process
        def worker(self, env, idx):
            sim.suspend()

    class IndexedModel(sim.Model):
        flow: IndexedWorker = IndexedWorker()

    with pytest.raises(ValueError, match="copy index"):
        IndexedModel()

    class Tag(sim.Struct):
        value: int

    class MisplacedView(sim.Component):
        worker: sim.Spawnable

        @sim.process
        def worker(self, env, tag: Tag, idx):
            sim.suspend()

    class MisplacedModel(sim.Model):
        flow: MisplacedView = MisplacedView()

    with pytest.raises(ValueError, match="last parameter"):
        MisplacedModel()

    class BadDefault(sim.Component):
        worker: sim.Spawnable = object()

    class BadDefaultModel(sim.Model):
        flow: BadDefault = BadDefault()

    with pytest.raises(ValueError, match="only Queue/Pool/Store"):
        BadDefaultModel()


def test_nested_component_declaration_and_namespace_errors_are_rejected():
    class Child(sim.Component):
        count: sim.State

    class MissingChild(sim.Component):
        child: Child

    class MissingModel(sim.Model):
        parent: MissingChild = MissingChild()

    with pytest.raises(ValueError, match="needs a Child instance default"):
        MissingModel()

    class WrongChild(sim.Component):
        child: Child = object()

    class WrongModel(sim.Model):
        parent: WrongChild = WrongChild()

    with pytest.raises(TypeError, match="default must be a Child instance"):
        WrongModel()

    class EmptyChildren(sim.Component):
        children: list[Child] = []

    class EmptyModel(sim.Model):
        parent: EmptyChildren = EmptyChildren()

    with pytest.raises(ValueError, match="non-empty"):
        EmptyModel()

    class Parent(sim.Component):
        child: Child = Child()

    class Network(sim.Model):
        parent: Parent = Parent()

    direct = Network()
    with pytest.raises(ValueError, match="cannot use env.parent.child"):
        @direct.process
        def direct_actor(env: Network):
            _ = env.parent.child
            sim.suspend()

    unknown = Network()
    with pytest.raises(ValueError, match="unknown component field"):
        @unknown.process
        def unknown_actor(env: Network):
            env.parent.child.missing = 1
            sim.suspend()

    dynamic = Network()
    with pytest.raises(ValueError, match="dynamic getattr"):
        @dynamic.process
        def dynamic_actor(env: Network):
            env.parent.child.count = getattr(env.parent.child, "count")
            sim.suspend()

    nested_field = Network()
    with pytest.raises(ValueError, match="below component field"):
        @nested_field.process
        def nested_actor(env: Network):
            _ = env.parent.child.count.extra
            sim.suspend()


def test_component_collection_declaration_errors_are_rejected():
    class Item(sim.Component):
        count: sim.State

    class Empty(sim.Model):
        items: list[Item] = []

    with pytest.raises(ValueError, match="non-empty"):
        Empty()

    class Wrong(sim.Model):
        items: list[Item] = [object()]

    with pytest.raises(TypeError, match="items must be Item instances"):
        Wrong()

    class MissingCount(sim.Component):
        lanes: sim.PQueues = sim.count("queue_count")

    class MissingCountModel(sim.Model):
        items: list[MissingCount] = [MissingCount()]

    with pytest.raises(ValueError, match="must name an int constant"):
        MissingCountModel()

    class MissingWorker(sim.Component):
        worker: sim.Processes

    class MissingWorkerModel(sim.Model):
        items: list[MissingWorker] = [MissingWorker()]

    with pytest.raises(ValueError, match="same-named @sim.process"):
        MissingWorkerModel()

    class MissingConst(sim.Component):
        count: sim.Const[int]

    class MissingConstModel(sim.Model):
        items: list[MissingConst] = [MissingConst()]

    with pytest.raises(ValueError, match="constant 'count'"):
        MissingConstModel()

    class BadConst(sim.Component):
        count: sim.Const[int]

        def __init__(self):
            self.count = 1.5

    class BadConstModel(sim.Model):
        item: BadConst = BadConst()

    with pytest.raises(ValueError, match="constant 'count' must be int"):
        BadConstModel()


def test_component_collection_namespace_errors_are_rejected():
    class Item(sim.Component):
        count: sim.State

        def __init__(self):
            self.values = [1]

    class Network(sim.Model):
        items: list[Item] = [Item()]

    direct = Network()
    with pytest.raises(ValueError, match="cannot use env.items directly"):
        @direct.process
        def direct_actor(env: Network):
            _ = env.items
            sim.suspend()

    item_alias = Network()
    with pytest.raises(ValueError, match=r"env.items\[\.\.\.\] directly"):
        @item_alias.process
        def item_alias_actor(env: Network):
            _ = env.items[0]
            sim.suspend()

    unknown = Network()
    with pytest.raises(ValueError, match="unknown component collection field"):
        @unknown.process
        def unknown_actor(env: Network):
            env.items[0].missing = 1
            sim.suspend()

    dynamic = Network()
    with pytest.raises(ValueError, match="dynamic getattr"):
        @dynamic.process
        def dynamic_actor(env: Network):
            env.items[0].count = getattr(env.items[0], "count")
            sim.suspend()

    unsupported_constant = Network()
    with pytest.raises(ValueError, match="unknown component collection field"):
        @unsupported_constant.process
        def unsupported_actor(env: Network):
            env.items[0].count = env.items[0].values[0]
            sim.suspend()

class Stage(sim.Component):
    seen: sim.State
    inbox: sim.Store
    outbox: sim.Store

    def __init__(self, inbox=None):
        if inbox is not None:
            self.inbox = inbox

    @sim.process
    def worker(self, env):
        while True:
            item = self.inbox.take()
            self.seen += 1
            self.outbox.put(item)


def test_component_store_wiring_shares_entity_and_runs():
    class Line(sim.Model):
        done: sim.Output
        first: Stage = Stage()
        second: Stage = Stage(inbox=first.outbox)

    model = Line()
    assert model.stores == {"first__inbox": None, "first__outbox": None,
                            "second__outbox": None}
    assert "second__inbox" not in model.dtype.fields
    second_decl = next(decl for decl in model._component_decls
                       if decl.name == "second")
    assert second_decl.direct_field_map["inbox"] == "first__outbox"
    assert second_decl.aliased_fields == ("inbox",)

    @model.process
    def feed(env: Line):
        env.first.inbox.put(7)

    @model.process
    def drain(env: Line):
        env.done = float(env.second.outbox.take())

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0, seed=3)
    assert exp.run() == 0
    assert exp["done"][0] == 7.0
    assert exp.trials["first__seen"][0] == 1
    assert exp.trials["second__seen"][0] == 1


def test_component_store_wiring_dag_keeps_entity_in_owner_block():
    class Line(sim.Model):
        first: Stage = Stage()
        second: Stage = Stage(inbox=first.outbox)

    graph = Line().process_dag()
    blocks = {block.name: block.members for block in graph.blocks}
    assert "store:first__outbox" in blocks["first"]
    assert "store:first__outbox" not in blocks["second"]
    assert "store:second__inbox" not in blocks["second"]
    edges = {(edge.source, edge.target, edge.label) for edge in graph.edges}
    assert ("store:first__outbox", "process:second__worker",
            "store_take") in edges


def test_component_resource_wiring_shares_resource():
    class Machinist(sim.Component):
        done: sim.Output
        machine: sim.Resource

        def __init__(self, machine=None):
            if machine is not None:
                self.machine = machine

        @sim.process
        def run(self, env):
            self.machine.acquire()
            sim.hold(1.0)
            self.machine.release()
            self.done = sim.now()

    class Shop(sim.Model):
        a: Machinist = Machinist()
        b: Machinist = Machinist(machine=a.machine)

    model = Shop()
    assert model.resources == ["a__machine"]
    exp = model.experiment(replications=1, duration=5.0, warmup=0.0, seed=3)
    assert exp.run() == 0
    assert {exp["a__done"][0], exp["b__done"][0]} == {1.0, 2.0}


def test_component_wiring_declaration_errors_are_rejected():
    class Mixed(sim.Component):
        inbox: sim.Store
        gate: sim.Resource

        def __init__(self, inbox=None):
            if inbox is not None:
                self.inbox = inbox

    with pytest.raises(ValueError, match="field kinds must match"):
        class KindMismatch(sim.Model):
            a: Mixed = Mixed()
            b: Mixed = Mixed(inbox=a.gate)

        KindMismatch()

    orphan = Stage()
    with pytest.raises(ValueError, match="not declared on the model"):
        class OrphanTarget(sim.Model):
            a: Stage = Stage(inbox=orphan.outbox)

        OrphanTarget()

    with pytest.raises(ValueError,
                       match="not supported for collections"):
        class WiredCollection(sim.Model):
            first: Stage = Stage()
            items: list[Stage] = [Stage(inbox=first.outbox), Stage()]

        WiredCollection()

    with pytest.raises(ValueError, match="collection item"):
        class WiredToCollection(sim.Model):
            items: list[Stage] = [Stage(), Stage()]
            a: Stage = Stage(inbox=items[0].outbox)

        WiredToCollection()

    shared = Stage()
    with pytest.raises(ValueError, match="ambiguous"):
        class Ambiguous(sim.Model):
            a: Stage = shared
            b: Stage = shared
            c: Stage = Stage(inbox=shared.outbox)

        Ambiguous()


def test_component_wiring_allows_forward_targets_and_chains():
    # Wiring is resolved after the whole tree is built, so a field may be
    # wired to a target declared later on the model.
    first_stage = Stage()
    second_stage = Stage(inbox=first_stage.outbox)
    third_stage = Stage(inbox=second_stage.inbox)

    class Line(sim.Model):
        third: Stage = third_stage
        second: Stage = second_stage
        first: Stage = first_stage

    model = Line()
    decls = {decl.name: decl for decl in model._component_decls}
    assert decls["second"].direct_field_map["inbox"] == "first__outbox"
    assert decls["third"].direct_field_map["inbox"] == "first__outbox"
    assert decls["second"].aliased_fields == ("inbox",)
    # Wired fields declare no entity of their own.
    assert "second__inbox" not in model.dtype.fields
    assert "third__inbox" not in model.dtype.fields


def test_component_wiring_cycle_is_rejected():
    left_stage, right_stage = Stage(), Stage()
    # Capture the field references before assigning, so each stage's inbox
    # names the other's inbox -- a cycle with no backing entity.
    left_inbox = left_stage.inbox
    right_inbox = right_stage.inbox
    left_stage.inbox = right_inbox
    right_stage.inbox = left_inbox

    class Cycle(sim.Model):
        left: Stage = left_stage
        right: Stage = right_stage

    with pytest.raises(ValueError, match="wiring cycle"):
        Cycle()


def test_component_field_access_outside_model_returns_wiring_ref():
    stage = Stage()
    ref = stage.outbox
    assert ref.instance is stage
    assert ref.field == "outbox"
    assert ref.kind == "store"
    with pytest.raises(AttributeError):
        stage.seen  # Output fields are not wirable
    with pytest.raises(AttributeError):
        stage.missing

class RefNode(sim.Component):
    got: sim.State
    inbox: sim.Store
    downstream: sim.Ref["RefNode"]

    def __init__(self, tag: int = 0, downstream=None):
        self.tag = tag
        if downstream is not None:
            self.downstream = downstream

    @sim.process
    def take(self, env):
        while True:
            self.inbox.take()
            self.got += 1


class RefRelay(sim.Component):
    passed: sim.State
    inbox: sim.Store
    downstream: sim.Ref[sim.Component]

    def __init__(self, downstream=None):
        if downstream is not None:
            self.downstream = downstream

    @sim.process
    def relay(self, env):
        while True:
            item = self.inbox.take()
            self.passed += 1
            self.downstream.inbox.put(item)


def test_component_ref_chain_supports_forward_declaration_order():
    second = RefRelay()
    end = RefNode()
    second.downstream = end

    class Line(sim.Model):
        relay1: RefRelay = RefRelay(downstream=second)
        relay2: RefRelay = second
        sink: RefNode = end

    model = Line()

    @model.process
    def feed(env: Line):
        env.relay1.inbox.put(1)
        env.relay1.inbox.put(2)

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0, seed=5)
    assert exp.run() == 0
    assert exp.trials["relay1__passed"][0] == 2
    assert exp.trials["relay2__passed"][0] == 2
    assert exp.trials["sink__got"][0] == 2


def test_component_refs_table_routes_by_condition():
    class Dispatcher(sim.Component):
        sent: sim.State
        inbox: sim.Store
        routes: sim.Refs[RefNode]

        def __init__(self, routes=()):
            self.routes = tuple(routes)

        @sim.process
        def route(self, env):
            while True:
                item = self.inbox.take()
                self.sent += 1
                self.routes[item % 3].inbox.put(item)

    class Shop(sim.Model):
        nodes: list[RefNode] = [RefNode(), RefNode(), RefNode()]
        dispatch: Dispatcher = Dispatcher(routes=(nodes[0], nodes[1],
                                                  nodes[2]))

    model = Shop()

    @model.process
    def feed(env: Shop):
        for i in range(7):
            env.dispatch.inbox.put(i)

    @model.process
    def probe(env: Shop):
        env.dispatch.routes[1].inbox.put(100)

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0, seed=5)
    assert exp.run() == 0
    assert list(exp.trials["nodes__got"][0]) == [3, 3, 2]
    assert exp.trials["dispatch__sent"][0] == 7


def test_component_collection_items_can_reference_mixed_targets():
    end = RefNode()
    first, second = RefRelay(), RefRelay()
    first.downstream = second   # item of the same collection
    second.downstream = end     # separately declared component

    class Chain(sim.Model):
        relays: list[RefRelay] = [first, second]
        sink: RefNode = end

    model = Chain()

    @model.process
    def feed(env: Chain):
        env.relays[0].inbox.put(9)

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0, seed=5)
    assert exp.run() == 0
    assert list(exp.trials["relays__passed"][0]) == [1, 1]
    assert exp.trials["sink__got"][0] == 1


def test_model_callback_can_follow_refs_with_dynamic_index():
    a, b, c = RefNode(), RefNode(), RefNode()
    a.downstream = b
    b.downstream = c
    c.downstream = a

    class Ring(sim.Model):
        nodes: list[RefNode] = [a, b, c]

    model = Ring()

    @model.process
    def probe(env: Ring):
        for j in range(3):
            env.nodes[j].downstream.inbox.put(j)

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0, seed=5)
    assert exp.run() == 0
    assert list(exp.trials["nodes__got"][0]) == [1, 1, 1]


def test_component_ref_exposes_target_constants():
    class Reader(sim.Component):
        tag_seen: sim.State
        downstream: sim.Ref[RefNode]

        def __init__(self, downstream=None):
            if downstream is not None:
                self.downstream = downstream

        @sim.process
        def read(self, env):
            self.tag_seen += self.downstream.tag

    class M(sim.Model):
        target: RefNode = RefNode(tag=7)
        reader: Reader = Reader(downstream=target)

    model = M()
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0, seed=5)
    assert exp.run() == 0
    assert exp.trials["reader__tag_seen"][0] == 7


def test_component_ref_declaration_errors_are_rejected():
    with pytest.raises(ValueError, match="only supported on Component"):
        class BadModel(sim.Model):
            x: sim.Ref[RefNode]

        BadModel()

    with pytest.raises(ValueError, match="no target for this instance"):
        class Unset(sim.Model):
            relay: RefRelay = RefRelay()

        Unset()

    with pytest.raises(ValueError, match="not declared on the model"):
        class Undeclared(sim.Model):
            relay: RefRelay = RefRelay(downstream=RefNode())
            sink: RefNode = RefNode()

        Undeclared()

    shared = RefNode()
    with pytest.raises(ValueError, match="ambiguous"):
        class Ambiguous(sim.Model):
            x: RefNode = shared
            y: RefNode = shared
            relay: RefRelay = RefRelay(downstream=shared)

        Ambiguous()

    with pytest.raises(TypeError, match="instance or None"):
        class WrongType(sim.Model):
            relay: RefRelay = RefRelay(downstream=5)

        WrongType()


def test_component_refs_table_errors_are_rejected():
    class Dispatcher(sim.Component):
        inbox: sim.Store
        routes: sim.Refs[RefNode]

        def __init__(self, routes=()):
            self.routes = tuple(routes)

        @sim.process
        def route(self, env):
            while True:
                item = self.inbox.take()
                self.routes[item % 2].inbox.put(item)

    with pytest.raises(ValueError, match="single component collection"):
        class SingleTarget(sim.Model):
            sink: RefNode = RefNode()
            dispatch: Dispatcher = Dispatcher(routes=(sink,))

        SingleTarget()

    with pytest.raises(ValueError, match="single component collection"):
        class MixedCollections(sim.Model):
            left: list[RefNode] = [RefNode(), RefNode()]
            right: list[RefNode] = [RefNode(), RefNode()]
            dispatch: Dispatcher = Dispatcher(routes=(left[0], right[0]))

        MixedCollections()


def test_component_ref_usage_errors_are_rejected():
    left, right = RefNode(), RefNode()
    left.downstream = right

    class Pair(sim.Model):
        a: RefNode = left
        b: RefNode = right

    bare = Pair()
    with pytest.raises(ValueError, match="cannot use env.a.downstream "
                                         "directly"):
        @bare.process
        def bare_ref(env: Pair):
            _ = env.a.downstream
            sim.suspend()

    class Router(sim.Component):
        inbox: sim.Store
        routes: sim.Refs[RefNode]

        def __init__(self, routes=()):
            self.routes = tuple(routes)

        @sim.process
        def route(self, env):
            self.inbox.take()

    class Routed(sim.Model):
        nodes: list[RefNode] = [RefNode(), RefNode()]
        router: Router = Router(routes=(nodes[0], nodes[1]))

    unindexed = Routed()
    with pytest.raises(ValueError, match="must index env.router.routes"):
        @unindexed.process
        def bare_table(env: Routed):
            _ = env.router.routes
            sim.suspend()

    out_of_range = Routed()
    with pytest.raises(ValueError, match="out of range"):
        @out_of_range.process
        def oob(env: Routed):
            env.router.routes[5].inbox.put(1)

    hetero_sink = RefNode()
    x, y = RefNode(), RefNode()
    x.downstream = y
    y.downstream = hetero_sink

    class Hetero(sim.Model):
        nodes: list[RefNode] = [x, y]
        sink: RefNode = hetero_sink

    hetero = Hetero()
    with pytest.raises(ValueError, match="same component declaration"):
        @hetero.process
        def dynamic(env: Hetero):
            for j in range(2):
                env.nodes[j].downstream.inbox.put(j)
