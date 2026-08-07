import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import cimba.sim as sim
from cimba._model import _callback_function_key


def test_callback_fingerprint_tracks_closure_and_array_values():
    def callback(value, table):
        def apply(number):
            return number + value + table[0]
        return apply

    table = np.array([3], dtype=np.int64)
    first = _callback_function_key(callback(2, table))
    assert first != _callback_function_key(callback(5, table))
    table[0] = 8
    assert first != _callback_function_key(callback(2, table))


def test_precompile_uses_real_model_and_exposes_immutable_plan(monkeypatch):
    monkeypatch.setenv("CIMBA_CACHE", "0")
    constructed = 0

    class Worker(sim.Component):
        count: sim.State

        @sim.function
        def value(self) -> int:
            return self.count

        @sim.process
        def run(self, env):
            self.count += 1

    class Network(sim.Model):
        worker: Worker = Worker()

        @sim.function
        def identity(self, value: int) -> int:
            return value

        def __init__(self):
            nonlocal constructed
            constructed += 1
            super().__init__()

    assert constructed == 0
    assert Network.compilation_status().state == "pending"

    Network()
    assert constructed == 1
    assert Network.compilation_status().state == "ready"
    plan = Network.compilation_plan()
    assert isinstance(plan, sim.CompilationPlan)
    assert plan.process_names == ("worker__run",)
    assert plan.function_names == ("model:identity", "worker__value")
    assert len(plan.function_keys) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.model_name = "changed"  # type: ignore[misc]


def test_lazy_and_explicit_precompile_modes(monkeypatch):
    monkeypatch.setenv("CIMBA_CACHE", "0")

    class Worker(sim.Component):
        count: sim.State

        @sim.process
        def run(self, env):
            self.count += 1

    class LazyNetwork(sim.Model):
        __cimba_precompile__ = "lazy"
        worker: Worker = Worker()

    lazy = LazyNetwork()
    assert LazyNetwork.compilation_status().state == "pending"
    lazy.experiment(replications=1, duration=1.0, warmup=0.0)
    assert LazyNetwork.compilation_status().state == "ready"

    class ExplicitNetwork(sim.Model):
        __cimba_precompile__ = "explicit"
        worker: Worker = Worker()

    explicit = ExplicitNetwork()
    explicit.experiment(replications=1, duration=1.0, warmup=0.0)
    assert ExplicitNetwork.compilation_status().state == "pending"
    status = ExplicitNetwork.precompile()
    assert status.state == "ready"


def test_precompile_failure_is_observable_and_retryable(monkeypatch):
    monkeypatch.setenv("CIMBA_CACHE", "0")
    global_name = "_CIMBA_LATE_COMPILATION_TEST_VALUE"
    monkeypatch.delitem(globals(), global_name, raising=False)

    class Worker(sim.Component):
        count: sim.State

        @sim.process
        def run(self, env):
            self.count += (  # type: ignore[name-defined]
                _CIMBA_LATE_COMPILATION_TEST_VALUE
            )

    class Network(sim.Model):
        worker: Worker = Worker()

    Network()
    failed = Network.compilation_status()
    assert failed.state == "failed"
    assert failed.error

    monkeypatch.setitem(globals(), global_name, 2)
    ready = Network.precompile()
    assert ready.state == "ready"
    assert ready.error is None


def test_persistent_callback_cache_hits_in_a_fresh_interpreter(tmp_path):
    cache = tmp_path / "callback-cache"
    command = [
        sys.executable,
        "-c",
        (
            "import json,tempfile; from pathlib import Path; "
            "from tutorial.tut_5_1 import build_model; "
            "m=build_model(Path(tempfile.mkdtemp())); "
            "e=m.experiment(replications=1,duration=1.0,warmup=0.0); "
            "failures=e.run(); s=type(m).compilation_status(); "
            "c=m.callback_cache_stats(); "
            "print(json.dumps({"
            "'state':s.state,'hits':s.cache_hits,"
            "'misses':s.cache_misses,'writes':s.cache_writes,"
            "'callback_hits':c.hits,'callback_misses':c.misses,"
            "'failures':failures}))"
        ),
    ]
    env = os.environ.copy()
    env["CIMBA_CACHE"] = "1"
    env["CIMBA_CACHE_DIR"] = str(cache)
    root = Path(__file__).resolve().parents[1]

    first = subprocess.run(
        command, cwd=root, env=env, check=True, capture_output=True, text=True)
    second = subprocess.run(
        command, cwd=root, env=env, check=True, capture_output=True, text=True)
    cold = json.loads(first.stdout)
    warm = json.loads(second.stdout)

    assert cold["state"] == warm["state"] == "ready"
    assert cold["failures"] == warm["failures"] == 0
    assert cold["misses"] > 0
    assert cold["writes"] > 0
    assert warm["hits"] > 0
    assert warm["misses"] < cold["misses"]
    # Every callback is class-declared, so the reusable class plan handles
    # compilation and the safe per-instance fallback has no work to do.
    assert cold["callback_hits"] == cold["callback_misses"] == 0
    assert warm["callback_hits"] == warm["callback_misses"] == 0


def test_indexed_callback_cache_key_includes_record_layout(tmp_path):
    script = tmp_path / "indexed_layout_cache.py"
    script.write_text(
        """
import json
import sys

import numpy as np

import cimba.sim as sim


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


annotations = {"sources": list[Source]}
if int(sys.argv[1]):
    annotations = {"extra": sim.State, **annotations}
Network = type(
    "Network",
    (sim.Model,),
    {
        "__module__": __name__,
        "__annotations__": annotations,
        "sources": [Source(), Source()],
    },
)
model = Network()
experiment = model.experiment(
    sources__demand=[
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([3.0, 4.0], dtype=np.float64),
    ],
    replications=1,
    duration=1.0,
    warmup=0.0,
    seed=41,
)
failures = experiment.run()
status = Network.compilation_status()
print(json.dumps({
    "failures": failures,
    "totals": experiment["sources__total"].tolist(),
    "hits": status.cache_hits,
    "misses": status.cache_misses,
}))
""".lstrip()
    )
    env = os.environ.copy()
    env["CIMBA_CACHE"] = "1"
    env["CIMBA_CACHE_DIR"] = str(tmp_path / "cache")
    root = Path(__file__).resolve().parents[1]

    results = []
    for has_extra_output in (1, 0):
        completed = subprocess.run(
            [sys.executable, str(script), str(has_extra_output)],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(completed.stdout))

    assert [result["failures"] for result in results] == [0, 0]
    assert [result["totals"] for result in results] == [
        [[3.0, 7.0]],
        [[3.0, 7.0]],
    ]
    # Generic lifecycle callbacks still hit, but the record-specific indexed
    # process must compile separately for the second layout.
    assert results[1]["hits"] > 0
    assert results[1]["misses"] > 0
