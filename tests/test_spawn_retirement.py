"""Retiring finished spawned processes costs the same however many are alive
(#27), and keeps the engine's exit semantics in the cases it can't skip."""

import time

import cimba as cb

INTERRUPTED = -2
TIMEOUT = -5


def run(model, **kwargs):
    result = cb.Experiment(model, **kwargs).run(workers=1)
    assert not result.failed.any(), result.failure_reasons
    return result[model]


def value(output):
    return output.values.tolist()[0][0]


class Token(cb.Model):
    @cb.process
    def flow(self):
        cb.hold(cb.random.exponential(1.0))


class Burst(cb.Model):
    size: cb.Param[int] = 1

    @cb.process
    def arrive(self):
        for _ in range(self.size):
            cb.spawn(Token)


def per_process(size):
    model = Burst()
    model.size = size
    experiment = cb.Experiment(model)
    experiment.run(workers=1)          # compiled
    best = float("inf")
    for _ in range(2):
        started = time.perf_counter()
        assert not experiment.run(workers=1).failed.any()
        best = min(best, time.perf_counter() - started)
    return best / size


def test_exit_cost_does_not_grow_with_the_live_processes():
    # Each exit used to sweep the event queue, which holds an event per live
    # process: 20 times the processes made each exit ~20 times dearer.
    assert per_process(40_000) < 4 * per_process(2_000)


class Held(cb.Model):
    """A token that ends holding a unit: the exit releases it to Waiter."""
    desk: cb.Ref["Office"]

    @cb.process
    def flow(self):
        self.desk.clerks.acquire()
        cb.hold(2.0)


class Office(cb.Model):
    clerks: cb.Resource = cb.Resource(capacity=1)
    waited_until: cb.Output[float]

    @cb.process
    def arrive(self):
        cb.spawn(Held, desk=self)

    @cb.process
    def waiter(self):
        cb.hold(1.0)
        self.clerks.acquire()
        self.waited_until = cb.now()


def test_a_token_ending_with_a_holding_releases_it():
    assert value(run(Office()).waited_until) == 2.0


class Timed(cb.Model):
    """A token that ends with a timer still set: the timer never fires."""
    owner: cb.Ref["Timers"]

    @cb.process
    def flow(self):
        cb.this_process().timer_set(5.0, TIMEOUT)
        cb.hold(1.0)


class Timers(cb.Model):
    done: cb.Output[float]

    @cb.process
    def arrive(self):
        for _ in range(100):
            cb.spawn(Timed, owner=self)
        cb.hold(10.0)
        self.done = cb.now()


def test_a_token_ending_with_a_timer_set_cancels_it():
    assert value(run(Timers()).done) == 10.0


class Target(cb.Model):
    owner: cb.Ref["Interrupter"]

    @cb.process
    def flow(self):
        self.owner.pointer = cb.this_process().pointer
        cb.hold(5.0)


class Interrupter(cb.Model):
    """Interrupts a token at the instant it finishes, before or after."""
    pointer: int = 0
    priority: cb.Param[int] = 0
    later: cb.Output[float]

    @cb.process
    def arrive(self):
        cb.spawn(Target, owner=self)
        for _ in range(50):
            cb.spawn(Token)
        cb.hold(5.0)
        cb.Process(self.pointer).interrupt(INTERRUPTED, self.priority)
        cb.hold(1.0)
        self.later = cb.now()


def test_an_interrupt_at_the_instant_a_token_ends_is_harmless():
    for priority in (-5, 0, 5):
        model = Interrupter()
        model.priority = priority
        assert value(run(model, replications=20).later) == 6.0


class Failing(cb.Model):
    clerks: cb.Resource = cb.Resource(capacity=1)
    bad: cb.Param[int] = 1

    @cb.process
    def arrive(self):
        for _ in range(500):
            cb.spawn(Token)
            cb.hold(0.01)

    @cb.process
    def culprit(self):
        cb.hold(3.0)
        if self.bad:
            self.clerks.release(1)


def test_abandoned_trials_with_retired_tokens_leak_nothing():
    model = Failing()
    results = cb.Experiment(model, replications=200).run(workers=1)
    assert results.failed.all()
    model.bad = 0
    assert not cb.Experiment(model, replications=50).run(workers=1).failed.any()
