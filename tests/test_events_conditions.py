"""Predicate and event methods dispatch through per-class callback tables."""

import cimba as cb


class Triggered(cb.Model):
    ready: cb.State[bool] = False
    gate: cb.Condition
    woke_at: cb.Output[float]

    @cb.predicate
    def is_ready(self):
        return self.ready

    @cb.event
    def activate(self):
        self.ready = True
        self.gate.signal()

    @cb.process
    def starter(self):
        cb.schedule(self.activate, 2.0)

    @cb.process
    def waiter(self):
        self.gate.wait_until(self.is_ready)
        self.woke_at = cb.now()


def test_event_signals_condition():
    model = Triggered()
    result = cb.Experiment(model, replications=3).run(workers=1)
    assert not result.failed.any()
    assert result[model].woke_at.values.tolist() == [[2.0, 2.0, 2.0]]


class Cancelled(cb.Model):
    fired: cb.State[bool] = False
    pending: cb.Output[bool]
    cancelled: cb.Output[bool]
    fired_out: cb.Output[bool]

    @cb.event
    def fire(self):
        self.fired = True

    @cb.process
    def start(self):
        event = cb.schedule(self.fire, 1.0)
        self.pending = event.pending()
        self.cancelled = event.cancel()

    @cb.on_end
    def finish(self):
        self.fired_out = self.fired


def test_scheduled_handle_cancellation():
    model = Cancelled()
    result = cb.Experiment(model).run(workers=1)
    assert result[model].pending.values[0, 0] == 1
    assert result[model].cancelled.values[0, 0] == 1
    assert result[model].fired_out.values[0, 0] == 0
