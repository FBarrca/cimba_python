"""Exhaustion policies, container initial levels and the window's end."""

import subprocess
import sys
import textwrap

import pytest

import cimba as cb
from cimba import inputs


class Reader(cb.Model):
    values: cb.Input[float]
    days: cb.Series[float] = cb.Series(step=1.0)
    use_series: cb.Param[bool] = False
    reads: cb.State[int] = 0
    count: cb.Output[int]
    ended_at: cb.Output[float]

    @cb.process
    def read(self):
        while True:
            cb.hold(1.0)
            if self.use_series:
                self.days.now()
            else:
                self.values.next()
            self.reads += 1

    @cb.on_end
    def finish(self):
        self.count = self.reads
        self.ended_at = cb.now()


# A sequence reads values 0, 1, 2 at t = 1, 2, 3 and runs out at t = 4. The
# series (origin 0) reads buckets 1, 2 at t = 1, 2 and runs out at t = 3.
@pytest.mark.parametrize("use_series, reads, ended_at",
                         [(False, 3.0, 4.0), (True, 2.0, 3.0)])
def test_end_trial_policy_ends_the_trial_successfully(use_series, reads, ended_at):
    model = Reader()
    model.use_series = use_series
    model.values = inputs.trace([1.0, 2.0, 3.0], on_exhausted="end_trial")
    model.days = inputs.trace([1.0, 2.0, 3.0], on_exhausted="end_trial")
    results = cb.Experiment(model, replications=2,
                            window=cb.Window(duration=10.0)).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].count.values.tolist() == [[reads, reads]]
    assert results[model].ended_at.values.tolist() == [[ended_at, ended_at]]


def test_fail_policy_still_fails_the_trial():
    model = Reader()
    model.values = inputs.trace([1.0, 2.0, 3.0])
    model.days = inputs.trace([0.0], on_exhausted="wrap")
    results = cb.Experiment(model, window=cb.Window(duration=10.0)).run(workers=1)
    assert results.failed.all()
    assert "exhausted after 3 values" in results.failure_reasons[0, 0]


class Stocked(cb.Model):
    stock: cb.Container = cb.Container(initial=5)
    at_start: cb.Output[int]
    at_end: cb.Output[int]

    @cb.on_start
    def look(self):
        self.at_start = self.stock.level()

    @cb.process
    def take(self):
        self.stock.get(2)

    @cb.on_end
    def finish(self):
        self.at_end = self.stock.level()


def test_container_initial_level():
    model = Stocked()
    results = cb.Experiment(model).run(workers=1)
    assert results[model].at_start.values.tolist() == [[5.0]]
    assert results[model].at_end.values.tolist() == [[3.0]]
    other = Stocked()
    other.stock = cb.Container(initial=0)
    assert cb.Experiment(other).run(workers=1)[other].at_start.values.tolist() == [[0.0]]
    with pytest.raises(ValueError):
        cb.Container(initial=-1)
    with pytest.raises(ValueError):
        cb.Container(initial=1.5)  # pyright: ignore[reportArgumentType]


class Sampler(cb.Model):
    samples: cb.Dataset
    count: cb.Output[int]

    @cb.process
    def sample(self):
        self.samples.record(0.0)
        cb.hold(1.0)
        self.samples.record(1.0)

    @cb.on_end
    def finish(self):
        self.count = self.samples.sample_count()


# Opening the window clears datasets. Without a warmup it opens before the
# processes start, so samples recorded at time 0 count.
@pytest.mark.parametrize("window, count",
                         [(cb.Window(duration=10.0), 2),
                          (cb.Window.until_idle(), 2),
                          (cb.Window(warmup=1.0, duration=10.0), 1)])
def test_window_opening_keeps_samples_from_its_start(window, count):
    model = Sampler()
    results = cb.Experiment(model, window=window).run(workers=1)
    assert results[model].count.values.tolist() == [[count]]


def test_window_end_stops_spawned_processes():
    # Before the fix this model never finished, so run it in a subprocess
    # with a timeout rather than hang the test session.
    script = textwrap.dedent("""
        import cimba as cb

        class Forever(cb.Model):
            owner: cb.Ref["Spawner"]

            @cb.process
            def loop(self):
                while True:
                    cb.hold(1.0)
                    self.owner.ticks += 1

        class Spawner(cb.Model):
            ticks: cb.State[int] = 0
            total: cb.Output[int]
            ended_at: cb.Output[float]

            @cb.process
            def go(self):
                cb.spawn(Forever, owner=self)
                while True:
                    cb.hold(1.0)

            @cb.on_end
            def finish(self):
                self.total = self.ticks
                self.ended_at = cb.now()

        model = Spawner()
        results = cb.Experiment(model, window=cb.Window(warmup=2.0, duration=5.0,
                                cooldown=3.0)).run(workers=1)
        assert not results.failed.any()
        print(results[model].ended_at[0, 0], results[model].total[0, 0])
    """)
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                               text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    ended_at, total = map(float, completed.stdout.split())
    assert ended_at == 10.0
    assert total <= 10.0
