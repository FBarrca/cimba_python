import pytest

import cimba.sim as sim


def test_model_callbacks_inherit_replace_remove_and_keep_declaration_order():
    class Base(sim.Model):
        value: sim.State
        total: sim.Output

        @sim.process
        def first(env: "Base"):
            env.value += 1

        @sim.process
        def replace_me(env: "Base"):
            env.value += 2

        @sim.collect
        def stats(env: "Base"):
            env.total = env.value

    class Derived(Base):
        @sim.process
        def replace_me(env: "Derived"):
            env.value += 20

        @sim.process
        def last(env: "Derived"):
            env.value += 100

    model = Derived()
    assert [process.name for process in model._processes] == [
        "first", "replace_me", "last",
    ]
    experiment = model.experiment(replications=1, duration=1.0, warmup=0.0)
    assert experiment.run() == 0
    assert experiment.results.total[0] == 121

    class Removed(Derived):
        def first(env):
            pass

        def stats(env):
            pass

    removed = Removed()
    assert [process.name for process in removed._processes] == [
        "replace_me", "last",
    ]
    assert removed._collect is None

    class DuplicateCollector(Derived):
        @sim.collect
        def another_stats(env: "DuplicateCollector"):
            env.total = env.value

    with pytest.raises(ValueError, match="multiple collect callbacks"):
        DuplicateCollector()


def test_callback_fields_validate_kind_binding_and_namespace_collisions():
    class WrongProcessField(sim.Model):
        ready: sim.Predicate

        @sim.process(field="ready")
        def worker(env):
            pass

    class WrongPredicateField(sim.Model):
        alarm: sim.Event

        @sim.predicate(field="alarm")
        def is_ready(env) -> bool:
            return True

    class WrongEventField(sim.Model):
        ready: sim.Predicate

        @sim.event(field="ready")
        def on_alarm(env):
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
        def ready(env) -> bool:
            return True

    class CollidingParam(sim.Model):
        worker: sim.Param

        @sim.process
        def worker(env):
            pass

    for cls in (CollidingPredicate, CollidingParam):
        with pytest.raises(ValueError, match="collides with a declared field"):
            cls()

    class DuplicateEventBinding(sim.Model):
        alarm: sim.Event

        @sim.event(field="alarm")
        def first_alarm(env):
            pass

        @sim.event(field="alarm")
        def second_alarm(env):
            pass

    with pytest.raises(ValueError, match="already bound"):
        DuplicateEventBinding()

    with pytest.raises(ValueError, match="spawnable processes cannot take"):
        class SpawnableField(sim.Model):
            workers: sim.Processes

            @sim.process(spawnable=True, field="workers")
            def worker(env):
                pass


def test_hidden_predicate_and_event_fields_are_generated_and_run():
    class Hidden(sim.Model):
        gate: sim.Condition
        opened: sim.State
        payload: sim.Output

        @sim.predicate
        def is_open(env: "Hidden") -> bool:
            return env.opened == 1

        @sim.event
        def alarm(env: "Hidden", data: int):
            env.payload = data

        @sim.process
        def driver(env: "Hidden"):
            env._ev_alarm.schedule(1.0, 42)
            env.opened = 1
            env.gate.signal()
            env.gate.wait_for(env._pred_is_open)
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
        def worker(env, first, second):
            pass

    class BadCollect(sim.Model):
        @sim.collect
        def stats(env, extra):
            pass

    class BadPredicateArgs(sim.Model):
        @sim.predicate
        def ready(env, extra) -> bool:
            return True

    class BadPredicateReturn(sim.Model):
        @sim.predicate
        def ready(env) -> int:
            return 1

    class BadEvent(sim.Model):
        @sim.event
        def alarm(env, data, extra):
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
        def experiment(env):
            pass

    with pytest.raises(ValueError, match="shadows a public sim.Model"):
        Protected()

    def callback(env):
        pass

    with pytest.raises(ValueError, match="cannot combine event"):
        sim.event(sim.predicate(callback))

    def function_event(self, env):
        pass

    with pytest.raises(ValueError, match="component function"):
        sim.function(sim.event(function_event))


def test_component_rejects_model_only_callback_markers_and_options():
    class Struct(sim.Struct):
        value: int

    class FieldComponent(sim.Component):
        @sim.process(field="workers")
        def worker(self, env):
            pass

    class StructComponent(sim.Component):
        @sim.process(struct=Struct)
        def worker(self, env):
            pass

    class PredicateComponent(sim.Component):
        @sim.predicate
        def ready(self, env) -> bool:
            return True

    for component in (FieldComponent(), StructComponent(), PredicateComponent()):
        class Owner(sim.Model):
            item: type(component) = component

        with pytest.raises(ValueError, match="component"):
            Owner()

    class StringCopies(sim.Model):
        @sim.process(copies="worker_count")
        def worker(env):
            pass

    with pytest.raises(TypeError, match="copies must be an int"):
        StringCopies()


def test_compilation_plan_covers_all_class_callback_categories_and_reuses():
    class Planned(sim.Model):
        ready: sim.Predicate
        alarm: sim.Event
        value: sim.State
        result: sim.Output

        @sim.predicate(field="ready")
        def is_ready(env: "Planned") -> bool:
            return env.value > 0

        @sim.event(field="alarm")
        def on_alarm(env: "Planned", data: int):
            env.value = data

        @sim.process
        def driver(env: "Planned"):
            env.alarm.schedule(0.0, 7)
            sim.hold(1.0)

        @sim.collect
        def stats(env: "Planned"):
            env.result = env.value

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
        def positive(env: "CallbackModel") -> bool:
            return env.value > 0

        @sim.event
        def set_value(env: "CallbackModel", data: int):
            env.value = data

        @sim.process
        def driver(env: "CallbackModel"):
            env._ev_set_value.schedule(0.0, 3)
            sim.hold(1.0)

        @sim.collect
        def stats(env: "CallbackModel"):
            env.result = env.value

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
