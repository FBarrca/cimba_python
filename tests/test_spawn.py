"""Dynamic model instances allocate trial-local records and run processes."""

import cimba as cb
from cimba import inputs


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
