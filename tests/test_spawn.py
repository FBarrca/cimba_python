"""Dynamic model instances allocate trial-local records and run processes."""

import pytest

import cimba as cb
from cimba import inputs
from cimba.compiler import ModelCompileError
from cimba.modeling import NotInCompiledCode


class Child(cb.Model):
    parent: cb.Ref["Parent"]
    value: cb.State[int]

    @cb.process
    def act(self):
        self.parent.total += self.value


class Parent(cb.Model):
    total: cb.State[int] = 0
    result: cb.Output[int]

    @cb.process
    def create(self):
        cb.spawn(Child, parent=self, value=5)
        cb.hold(1.0)

    @cb.on_end
    def finish(self):
        self.result = self.total


def test_spawned_model_runs_trial_local_process():
    parent = Parent()
    result = cb.Experiment(parent, replications=3).run(workers=1)
    assert result[parent].result.values.tolist() == [[5.0, 5.0, 5.0]]


class Order(cb.Model):
    warehouse: cb.Ref["Warehouse"]
    quantity: cb.State[int]

    @cb.process
    def deliver(self):
        self.warehouse.orders.put(self)


class Warehouse(cb.Model):
    orders: cb.Store[Order]
    received: cb.Output[int]

    @cb.process
    def create(self):
        cb.spawn(Order, warehouse=self, quantity=7)

    @cb.process
    def receive(self):
        order = self.orders.get()
        self.received = order.quantity


def test_spawned_model_flows_through_typed_store():
    warehouse = Warehouse()
    result = cb.Experiment(warehouse, replications=2).run(workers=1)
    assert result[warehouse].received.values.tolist() == [[7.0, 7.0]]


class CancelledChild(cb.Model):
    parent: cb.Ref["ReleaseParent"]

    @cb.process
    def act(self):
        cb.hold(0.5)
        self.parent.total += 1


class ReleaseParent(cb.Model):
    total: cb.State[int] = 0
    result: cb.Output[int]

    @cb.process
    def create(self):
        child = cb.spawn(CancelledChild, parent=self)
        cb.release(child)
        cb.hold(1.0)

    @cb.on_end
    def finish(self):
        self.result = self.total


def test_release_stops_spawned_process():
    parent = ReleaseParent()
    result = cb.Experiment(parent, replications=2).run(workers=1)
    assert result[parent].result.values.tolist() == [[0.0, 0.0]]


class FailingSpawned(cb.Model):
    parent: cb.Ref["FailureParent"]

    @cb.process
    def fail(self):
        cb.hold(self.parent.source.next())
        cb.hold(self.parent.source.next())


class FailureParent(cb.Model):
    source: cb.Input[float] = inputs.trace([1.0])

    @cb.process
    def create(self):
        cb.spawn(FailingSpawned, parent=self)


def test_spawned_model_cleanup_after_abandon_and_next_run():
    parent = FailureParent()
    experiment = cb.Experiment(parent, replications=3,
                               window=cb.Window(duration=5))
    failed = experiment.run(workers=1)
    assert failed.failed.all()
    recovered = experiment.run(workers=2)
    assert recovered.failed.all()
    assert all("source exhausted" in reason
               for reason in recovered.failure_reasons[0])


class LifecycleDocument(cb.Model):
    closed: cb.State[bool] = False
    dynamic_at_start: cb.State[bool] = False
    ran: cb.State[bool] = False

    @cb.on_start
    def start(self):
        self.dynamic_at_start = cb.is_dynamic(self)

    @cb.function
    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if cb.is_dynamic(self):
            cb.release(self)

    @cb.process
    def act(self):
        cb.hold(0.5)
        self.ran = True


class SpecializedDocument(LifecycleDocument):
    value: cb.Param[int] = 7


class LifecycleProbe(cb.Model):
    document: LifecycleDocument
    reference: cb.Ref[LifecycleDocument]
    documents: cb.Store[LifecycleDocument]
    static_at_start: cb.Output[bool]
    static_child_at_start: cb.Output[bool]
    static_at_end: cb.Output[bool]
    static_in_process: cb.Output[bool]
    dynamic_at_start: cb.Output[bool]
    dynamic_in_process: cb.Output[bool]
    dynamic_after_release: cb.Output[bool]
    static_ran: cb.Output[bool]
    dynamic_ran: cb.Output[bool]
    both_closed: cb.Output[bool]

    def __init__(self):
        self.document = SpecializedDocument()
        self.reference = self.document

    @cb.on_start
    def start(self):
        self.static_at_start = cb.is_dynamic(self)

    @cb.process
    def inspect(self):
        document = cb.spawn(SpecializedDocument)
        self.static_child_at_start = self.document.dynamic_at_start
        self.dynamic_at_start = document.dynamic_at_start
        self.documents.put(self.reference)
        self.documents.put(document)
        static = self.documents.get()
        dynamic = self.documents.get()
        self.static_in_process = cb.is_dynamic(static)
        self.dynamic_in_process = cb.is_dynamic(dynamic)
        static.close()
        static.close()
        dynamic.close()
        dynamic.close()
        cb.hold(1.0)
        self.dynamic_after_release = cb.is_dynamic(dynamic)
        self.static_ran = static.ran
        self.dynamic_ran = dynamic.ran
        self.both_closed = static.closed and dynamic.closed

    @cb.on_end
    def finish(self):
        self.static_at_end = cb.is_dynamic(self.reference)


@pytest.mark.parametrize("workers", [1, 2])
def test_is_dynamic_reports_creation_mode_through_lifecycle(workers):
    model = LifecycleProbe()
    experiment = cb.Experiment(model, replications=4)
    for _ in range(2):
        result = experiment.run(workers=workers)
        assert not result.failed.any(), result.failure_reasons
        for name in ("static_at_start", "static_child_at_start", "static_at_end",
                     "static_in_process", "dynamic_ran"):
            assert not getattr(result[model], name).values.any(), name
        for name in ("dynamic_at_start", "dynamic_in_process",
                     "dynamic_after_release", "static_ran", "both_closed"):
            assert getattr(result[model], name).values.all(), name


def test_is_dynamic_requires_compiled_code():
    with pytest.raises(NotInCompiledCode, match="is_dynamic"):
        cb.is_dynamic(LifecycleDocument())


@pytest.mark.parametrize("entity", [False, True])
def test_is_dynamic_rejects_non_models(entity):
    class Invalid(cb.Model):
        queue: cb.Store[int]

        @cb.process
        def inspect(self):
            if entity:
                cb.is_dynamic(self.queue)
            else:
                cb.is_dynamic(1)

    with pytest.raises(ModelCompileError, match="is_dynamic"):
        cb.Experiment(Invalid()).run(workers=1)
