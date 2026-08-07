import pytest

import cimba.sim as sim


def test_model_callbacks_inherit_replace_remove_and_keep_declaration_order():
    class Base(sim.Model):
        value: sim.State
        total: sim.Output

        @sim.process
        def first(self: "Base"):
            self.value += 1

        @sim.process
        def replace_me(self: "Base"):
            self.value += 2

        @sim.collect
        def stats(self: "Base"):
            self.total = self.value

    class Derived(Base):
        @sim.process
        def replace_me(self: "Derived"):
            self.value += 20

        @sim.process
        def last(self: "Derived"):
            self.value += 100

    model = Derived()
    assert [process.name for process in model._processes] == [
        "first", "replace_me", "last",
    ]
    experiment = model.experiment(replications=1, duration=1.0, warmup=0.0)
    assert experiment.run() == 0
    assert experiment.results.total[0] == 121

    class Removed(Derived):
        def first(self):
            pass

        def stats(self):
            pass

    removed = Removed()
    assert [process.name for process in removed._processes] == [
        "replace_me", "last",
    ]
    assert removed._collect is None

    class DuplicateCollector(Derived):
        @sim.collect
        def another_stats(self):
            self.total = self.value

    with pytest.raises(ValueError, match="multiple collect callbacks"):
        DuplicateCollector()


def test_callback_fields_validate_kind_binding_and_namespace_collisions():
    class WrongProcessField(sim.Model):
        ready: sim.Predicate

        @sim.process(field="ready")
        def worker(self):
            pass

    class WrongPredicateField(sim.Model):
        alarm: sim.Event

        @sim.predicate(field="alarm")
        def is_ready(self) -> bool:
            return True

    class WrongEventField(sim.Model):
        ready: sim.Predicate

        @sim.event(field="ready")
        def on_alarm(self):
            pass

    for cls, match in (
        (WrongProcessField, "sim.Processes field"),
        (WrongPredicateField, "sim.Predicate field"),
        (WrongEventField, "sim.Event field"),
    ):
        with pytest.raises(ValueError, match=match):
            cls()

    class CollidingPredicate(sim.Model):
        ready: sim.Predicate

        @sim.predicate
        def ready(self) -> bool:
            return True

    class CollidingParam(sim.Model):
        worker: sim.Param

        @sim.process
        def worker(self):
            pass

    for cls in (CollidingPredicate, CollidingParam):
        with pytest.raises(ValueError, match="collides with a declared field"):
            cls()

    class DuplicateEventBinding(sim.Model):
        alarm: sim.Event

        @sim.event(field="alarm")
        def first_alarm(self):
            pass

        @sim.event(field="alarm")
        def second_alarm(self):
            pass

    with pytest.raises(ValueError, match="already bound"):
        DuplicateEventBinding()

    with pytest.raises(ValueError, match="spawnable processes cannot take"):
        class SpawnableField(sim.Model):
            workers: sim.Processes

            @sim.process(spawnable=True, field="workers")
            def worker(self):
                pass


def test_hidden_predicate_and_event_fields_are_generated_and_run():
    class Hidden(sim.Model):
        gate: sim.Condition
        opened: sim.State
        payload: sim.Output

        @sim.predicate
        def is_open(self) -> bool:
            return self.opened == 1

        @sim.event
        def alarm(self, data: int):
            self.payload = data

        @sim.process
        def driver(self):
            self._ev_alarm.schedule(1.0, 42)
            self.opened = 1
            self.gate.signal()
            self.gate.wait_for(self._pred_is_open)
            sim.hold(2.0)

    model = Hidden()
    assert "_pred_is_open" in model.dtype.fields
    assert "_ev_alarm" in model.dtype.fields
    experiment = model.experiment(replications=1, duration=3.0, warmup=0.0)
    assert experiment.run() == 0
    assert experiment.results.payload[0] == 42


def test_model_callback_signature_protected_name_and_marker_errors():
    class BadProcess(sim.Model):
        @sim.process
        def worker(self, first, second):
            pass

    class BadCollect(sim.Model):
        @sim.collect
        def stats(self, extra):
            pass

    class BadPredicateArgs(sim.Model):
        @sim.predicate
        def ready(self, extra) -> bool:
            return True

    class BadPredicateReturn(sim.Model):
        @sim.predicate
        def ready(self) -> int:
            return 1

    class BadEvent(sim.Model):
        @sim.event
        def alarm(self, data, extra):
            pass

    cases = (
        (BadProcess, "process functions take"),
        (BadCollect, "collect functions take"),
        (BadPredicateArgs, "predicate functions take"),
        (BadPredicateReturn, "must return bool"),
        (BadEvent, "event functions take"),
    )
    for cls, match in cases:
        with pytest.raises(ValueError, match=match):
            cls()

    class Protected(sim.Model):
        @sim.process
        def experiment(self):
            pass

    with pytest.raises(ValueError, match="shadows a public sim.Model"):
        Protected()

    def callback(env):
        pass

    with pytest.raises(ValueError, match="cannot combine event"):
        sim.event(sim.predicate(callback))

    def function_event(self, env):
        pass

    with pytest.raises(ValueError, match="combine function"):
        sim.function(sim.event(function_event))


def test_component_supports_callback_fields_structs_predicates_and_events():
    class Struct(sim.Struct):
        value: int

    class Item(sim.Component):
        workers: sim.Processes
        ready: sim.Predicate
        alarm: sim.Event
        gate: sim.Condition
        value: sim.State
        result: sim.Output

        @sim.predicate(field="ready")
        def is_ready(self, env) -> bool:
            return self.value >= 3

        @sim.event(field="alarm")
        def on_alarm(self, env, data):
            self.value += data

        @sim.process(field="workers", struct=Struct)
        def worker(self, env):
            self.alarm.schedule(0.1, 3)
            sim.hold(0.2)
            self.gate.signal()
            sim.suspend()

        @sim.process
        def waiter(self, env):
            self.gate.wait_for(self.ready)
            self.result = self.value
            sim.suspend()

    class Hidden(sim.Component):
        gate: sim.Condition
        value: sim.State
        result: sim.Output

        @sim.predicate
        def positive(self, env) -> bool:
            return self.value > 0

        @sim.event
        def bump(self, env):
            self.value += 1

        @sim.process
        def driver(self, env):
            self._ev_bump.schedule(0.1)
            sim.hold(0.2)
            self.gate.signal()
            sim.suspend()

        @sim.process
        def waiter(self, env):
            self.gate.wait_for(self._pred_positive)
            self.result = self.value
            sim.suspend()

    class Owner(sim.Model):
        item: Item = Item()
        hidden: Hidden = Hidden()

    model = Owner()
    exp = model.experiment(replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp["item__result"][0] == 3.0
    assert exp["hidden__result"][0] == 1.0
    assert exp.trials["item__workers"][0] != 0

    # Symbolic process copies remain a nested-owner feature.
    class StringCopies(sim.Model):
        @sim.process(copies="worker_count")
        def worker(self):
            pass

    with pytest.raises(TypeError, match="copies must be an int"):
        StringCopies()


def test_model_functions_are_shared_root_helpers():
    class Policy(sim.Component):
        factor: sim.Param = 3.0

        @sim.function
        def scale(self, value: float) -> float:
            return self.factor * value

    class Worker(sim.Component):
        result: sim.Output

        @sim.process
        def run(self, env):
            self.result = env.adjust(2.0)

    class System(sim.Model):
        base: sim.Param = 4.0
        result: sim.Output
        policy: Policy = Policy()
        worker: Worker = Worker()

        @sim.function
        def adjust(self, value: float) -> float:
            return self.policy.scale(value) + self.base

        @sim.function
        def twice(self, value: float) -> float:
            return self.adjust(value) * 2.0

        @sim.process
        def calculate(self):
            self.result = self.twice(1.0)

    model = System()
    exp = model.experiment(replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp["result"][0] == 14.0
    assert exp["worker__result"][0] == 10.0
    edges = {
        (edge.source, edge.target, edge.label)
        for edge in model.process_dag().edges
    }
    assert (
        "process:calculate", "function:model:twice", "call"
    ) in edges
    assert (
        "function:model:twice", "function:model:adjust", "call"
    ) in edges
    assert (
        "function:model:adjust", "function:policy__scale", "call"
    ) in edges
    assert (
        "process:worker__run", "function:model:adjust", "call"
    ) in edges


def test_component_collection_predicates_and_events_bind_per_instance():
    class Cell(sim.Component):
        threshold: sim.Const[int]
        ready: sim.Predicate
        alarm: sim.Event
        gate: sim.Condition
        value: sim.State
        result: sim.Output

        def __init__(self, threshold: int):
            self.threshold = threshold

        @sim.predicate(field="ready")
        def is_ready(self, env) -> bool:
            return self.value >= self.threshold

        @sim.event(field="alarm")
        def on_alarm(self, env, data):
            self.value += data

        @sim.process
        def driver(self, env):
            self.alarm.schedule(0.1, self.threshold)
            sim.hold(0.2)
            self.gate.signal()
            sim.suspend()

        @sim.process
        def waiter(self, env):
            self.gate.wait_for(self.ready)
            self.result = self.value
            sim.suspend()

    class Network(sim.Model):
        cells: list[Cell] = [Cell(2), Cell(5)]

    model = Network()
    exp = model.experiment(replications=1, duration=1.0)
    assert exp.run() == 0
    assert exp["cells__result"][0].tolist() == [2.0, 5.0]
    assert exp.trials["cells__ready"][0].shape == (2,)
    assert exp.trials["cells__alarm"][0].shape == (2,)


def test_shared_callback_owner_validation():
    class MultipleCollectors(sim.Component):
        @sim.collect
        def first(self, env):
            pass

        @sim.collect
        def second(self, env):
            pass

    class MultipleCollectorsModel(sim.Model):
        item: MultipleCollectors = MultipleCollectors()

    with pytest.raises(ValueError, match="multiple collect callbacks"):
        MultipleCollectorsModel()

    class MissingBinding(sim.Component):
        @sim.process(field="workers")
        def run(self, env):
            pass

    class MissingBindingModel(sim.Model):
        item: MissingBinding = MissingBinding()

    with pytest.raises(ValueError, match="declared sim.Processes field"):
        MissingBindingModel()

    class WrongBinding(sim.Component):
        workers: sim.Event

        @sim.process(field="workers")
        def run(self, env):
            pass

    class WrongBindingModel(sim.Model):
        item: WrongBinding = WrongBinding()

    with pytest.raises(ValueError, match="declared sim.Processes field"):
        WrongBindingModel()

    class First(sim.Struct):
        value: int

    class Second(sim.Struct):
        value: int

    class StructMismatch(sim.Component):
        @sim.process(struct=First)
        def run(self, env, view: Second):
            pass

    class StructMismatchModel(sim.Model):
        item: StructMismatch = StructMismatch()

    with pytest.raises(ValueError, match="struct=.*annotation disagree"):
        StructMismatchModel()


def test_model_function_validation_and_recursion():
    class Recursive(sim.Model):
        @sim.function
        def first(self, value: float) -> float:
            return self.second(value)

        @sim.function
        def second(self, value: float) -> float:
            return self.first(value)

    with pytest.raises(ValueError, match="recursive model function call"):
        Recursive()

    class Mutation(sim.Model):
        value: sim.State

        @sim.function
        def bad(self) -> float:
            self.value = 1
            return 1.0

    with pytest.raises(ValueError, match="cannot mutate"):
        Mutation()

    class EntityOperation(sim.Model):
        queue: sim.Queue

        @sim.function
        def bad(self) -> float:
            self.queue.put(1)
            return 1.0

    with pytest.raises(ValueError, match="entity or runtime operation"):
        EntityOperation()

    class Protected(sim.Model):
        @sim.function
        def experiment(self) -> float:
            return 1.0

    with pytest.raises(ValueError, match="shadows a public sim.Model"):
        Protected()


def test_compilation_plan_covers_all_class_callback_categories_and_reuses():
    class Planned(sim.Model):
        ready: sim.Predicate
        alarm: sim.Event
        value: sim.State
        result: sim.Output

        @sim.predicate(field="ready")
        def is_ready(self) -> bool:
            return self.value > 0

        @sim.event(field="alarm")
        def on_alarm(self, data: int):
            self.value = data

        @sim.process
        def driver(self):
            self.alarm.schedule(0.0, 7)
            sim.hold(1.0)

        @sim.collect
        def stats(self):
            self.result = self.value

    first = Planned()
    plan = Planned.compilation_plan()
    assert plan is not None
    assert plan.process_names == ("driver",)
    assert plan.predicate_names == ("is_ready",)
    assert plan.event_names == ("on_alarm",)
    assert len(plan.collect_keys) == 1
    assert plan.callback_count == 13

    first_callbacks = first._aot_class_callbacks()
    second = Planned()
    second_callbacks = second._aot_class_callbacks()
    assert first_callbacks == second_callbacks

    experiment = second.experiment(replications=1, duration=2.0, warmup=0.0)
    assert experiment.run() == 0
    assert experiment.results.result[0] == 7


def test_model_callbacks_respect_lazy_and_explicit_precompile_modes():
    class CallbackModel(sim.Model):
        value: sim.State
        result: sim.Output

        @sim.predicate
        def positive(self) -> bool:
            return self.value > 0

        @sim.event
        def set_value(self, data: int):
            self.value = data

        @sim.process
        def driver(self):
            self._ev_set_value.schedule(0.0, 3)
            sim.hold(1.0)

        @sim.collect
        def stats(self):
            self.result = self.value

    class Lazy(CallbackModel):
        __cimba_precompile__ = "lazy"

    lazy = Lazy()
    assert Lazy.compilation_status().state == "pending"
    lazy_experiment = lazy.experiment(
        replications=1, duration=2.0, warmup=0.0)
    assert Lazy.compilation_status().state == "ready"
    assert lazy_experiment.run() == 0
    assert lazy_experiment.results.result[0] == 3

    class Explicit(CallbackModel):
        __cimba_precompile__ = "explicit"

    explicit = Explicit()
    explicit.experiment(replications=1, duration=2.0, warmup=0.0)
    assert Explicit.compilation_status().state == "pending"
    assert Explicit.precompile().state == "ready"


def test_removed_instance_callback_api_and_callback_free_direct_model():
    model = sim.Model("plain", outputs=["value"])
    for name in ("process", "collect", "predicate", "event"):
        assert not hasattr(model, name)
    assert sim.Model.compilation_status().state == "unavailable"
