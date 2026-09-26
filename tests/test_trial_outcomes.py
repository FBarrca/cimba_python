"""Trial validity is independent of values, and every run starts fresh."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import cimba.sim as sim
from cimba import _bindings as native


class Outcome(sim.Model):
    mode: sim.Param
    count: sim.State
    floating: sim.FloatState
    first: sim.Output
    second: sim.Output
    gate: sim.Condition

    @sim.predicate
    def ready(self) -> bool:
        raise ValueError("predicate failed")

    @sim.event
    def alarm(self):
        raise ValueError("event failed")

    @sim.function
    def broken(self, value: float) -> float:
        if value:
            raise ValueError("helper failed")
        return value

    @sim.process
    def run_trial(self):
        self.count += 1
        self.floating += 0.5
        self.first = self.count
        self.second = self.floating
        if self.mode == 1:
            native.trial_abandon()
        elif self.mode == 2:
            raise ValueError("process failed")
        elif self.mode == 3:
            self.gate.wait_for(self._pred_ready)
        elif self.mode == 4:
            self._ev_alarm.schedule(0.0)
        elif self.mode == 5:
            self.first = np.nan
        elif self.mode == 6:
            self.first = self.broken(1.0)

    @sim.process
    def wake_predicate(self):
        if self.mode == 3:
            sim.hold(0.0)
            self.gate.signal()

    @sim.collect
    def report(self):
        if self.mode == 7:
            raise ValueError("collector failed")


class BrokenCell(sim.Component):
    value: sim.Output

    @sim.process
    def work(self, env):
        self.value = 1.0
        raise ValueError("indexed process failed")


class IndexedFailure(sim.Model):
    cells: list[BrokenCell] = [BrokenCell(), BrokenCell()]


class NoOutputs(sim.Model):
    @sim.process
    def fail(self):
        native.trial_abandon()


def _check_recovery():
    # Force worker reuse after errors, independently of the host's CPU count.
    import cimba
    assert cimba.use_threads(1) == 1
    experiment = Outcome().experiment(
        mode=np.arange(8), replications=3, duration=None, warmup=0, seed=9)
    expected = np.repeat([False, True, True, True, True, False, True, True], 3)
    for workers in (1, 2, 1):
        assert cimba.use_threads(workers) == workers
        assert experiment.run() == int(expected.sum())
        np.testing.assert_array_equal(experiment.failed, expected)
        assert np.isnan(experiment["first"][expected]).all()
        assert np.isnan(experiment["second"][expected]).all()
        np.testing.assert_array_equal(experiment["first"][:3], 1.0)
        np.testing.assert_array_equal(experiment["second"][~expected], 0.5)
        table = experiment.summary()
        assert np.isnan(table["first"][[1, 2, 3, 4, 5, 6, 7]]).all()
        assert np.isnan(table["second"][[1, 2, 3, 4, 6, 7]]).all()
        assert table["second"][5] == 0.5
    for model in [IndexedFailure(), NoOutputs()]:
        experiment = model.experiment(replications=4, duration=None, warmup=0)
        assert experiment.run() == 4
        assert experiment.failed.all()
        assert experiment.run() == 4


def test_native_and_compiled_exception_recovery():
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Exception ignored" not in result.stderr


def test_failed_is_unavailable_before_execution():
    experiment = Outcome().experiment(mode=0, duration=None, warmup=0)
    with pytest.raises(RuntimeError, match="run"):
        _ = experiment.failed


if __name__ == "__main__":
    _check_recovery()
