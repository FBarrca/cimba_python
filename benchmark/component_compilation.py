"""Fresh-process benchmark for Cimba class-callback compilation.

The parent process intentionally imports no Cimba modules.  Every sample is
collected in a new interpreter so import, class planning, model construction,
first-experiment compilation, and cached experiment construction remain
separate measurements.

Examples::

    uv run python benchmark/component_compilation.py
    uv run python benchmark/component_compilation.py --runs 9 --json out.json
    uv run python benchmark/component_compilation.py --cache warm
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PHASES = (
    "cimba_import",
    "model_definition",
    "model_build",
    "first_experiment",
    "cached_experiment",
)


def _timed(operation):
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def _worker(scenario: str, scale: int, workdir: Path) -> dict[str, Any]:
    phases: dict[str, float] = {}

    def import_cimba():
        import cimba.sim  # noqa: F401

    _, phases["cimba_import"] = _timed(import_cimba)

    if scenario == "assembly-line":
        def define_assembly():
            from tutorial import tut_5_1
            return tut_5_1

        module, phases["model_definition"] = _timed(define_assembly)
        raw_dir = workdir / "assembly-line"
        raw_dir.mkdir(parents=True, exist_ok=True)
        model, phases["model_build"] = _timed(
            lambda: module.build_model(raw_dir))
        model_type = type(model)
        nodes = 4
        instances = 4
    elif scenario == "amusement-park":
        def define_park():
            from tutorial import tut_3_1
            return tut_3_1

        module, phases["model_definition"] = _timed(define_park)

        def build():
            return module.Park()

        model, phases["model_build"] = _timed(build)
        model_type = type(model)
        nodes = 3
        instances = 11
    elif scenario == "synthetic":
        # Dynamic class definition and construction are separated by the
        # helper's own timings below only for parity with tutorial models.
        # Definition includes the class-level component graph construction.
        import cimba.sim as sim
        # ``from __future__ import annotations`` makes the nested class's
        # marker annotations resolve through this module's globals.
        globals()["sim"] = sim

        class Payload(sim.Struct):
            value: float
            visits: int

        class Cell(sim.Component):
            count: sim.State
            queue: sim.Queue
            total: sim.Output

            @sim.process(copies=2, priority=3)
            def service(self, env, index):
                self.count += index + 1
                sim.suspend()

            @sim.collect
            def report(self, env):
                self.total = self.count + self.queue.level()

        def define_synthetic():
            if scale == 1:
                @sim.predicate(field="ready")
                def is_ready(self) -> bool:
                    return self.cells__count >= 0

                @sim.event(field="tick")
                def handle_tick(self, amount):
                    self.cells__count += amount

                @sim.collect
                def totals(self):
                    self.result = self.cells__count
            else:
                @sim.predicate(field="ready")
                def is_ready(self) -> bool:
                    return self.cells__count[0] >= 0

                @sim.event(field="tick")
                def handle_tick(self, amount):
                    self.cells__count[0] += amount

                @sim.collect
                def totals(self):
                    self.result = self.cells__count.sum()

            @sim.process(struct=Payload)
            def driver(self):
                sim.suspend()

            @sim.process(spawnable=True, struct=Payload)
            def dynamic(self):
                sim.suspend()

            namespace = {
                "__module__": __name__,
                "__annotations__": {
                    "cells": list[Cell],
                    "gate": sim.Condition,
                    "ready": sim.Predicate,
                    "tick": sim.Event,
                    "result": sim.Output,
                },
                "cells": [Cell() for _ in range(scale)],
                "is_ready": is_ready,
                "handle_tick": handle_tick,
                "driver": driver,
                "dynamic": dynamic,
                "totals": totals,
            }
            return type(
                f"SyntheticComponents{scale}", (sim.Model,), namespace)

        model_type, phases["model_definition"] = _timed(define_synthetic)

        def build_synthetic():
            return model_type(f"synthetic-{scale}")

        model, phases["model_build"] = _timed(build_synthetic)
        nodes = 1
        instances = scale
    else:  # pragma: no cover - argparse prevents this in normal use
        raise ValueError(f"unknown scenario: {scenario}")

    def experiment():
        return model.experiment(
            replications=1,
            duration=1.0,
            warmup=0.0,
            cooldown=0.0,
            seed=7,
        )

    first, phases["first_experiment"] = _timed(experiment)
    cached, phases["cached_experiment"] = _timed(experiment)
    if first.trials.dtype != cached.trials.dtype:
        raise RuntimeError("cached experiment changed the trial dtype")

    status = getattr(model_type, "compilation_status", lambda: None)()
    callback_cache = model.callback_cache_stats()
    import cimba
    return {
        "scenario": scenario,
        "scale": scale,
        "nodes": nodes,
        "instances": instances,
        "phases": phases,
        "precompile": (
            None if status is None else {
                "state": status.state,
                "seconds": status.seconds,
                "cache_hits": status.cache_hits,
                "cache_misses": status.cache_misses,
            }
        ),
        "callback_cache": {
            "hits": callback_cache.hits,
            "misses": callback_cache.misses,
            "writes": callback_cache.writes,
        },
        "metadata": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "cimba": cimba.__version__,
            "numba": importlib.metadata.version("numba"),
            "llvmlite": importlib.metadata.version("llvmlite"),
            "numpy": importlib.metadata.version("numpy"),
        },
    }


def _summary(values: list[float]) -> dict[str, float]:
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    ordered = sorted(values)
    return {
        "median": median,
        "mad": statistics.median(deviations),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _sample(
    scenario: str,
    scale: int,
    runs: int,
    cache_mode: str,
) -> dict[str, Any]:
    samples = []
    with tempfile.TemporaryDirectory(prefix="cimba-component-bench-") as root:
        root_path = Path(root)
        shared_cache = root_path / "cache"
        first_run = -1 if cache_mode == "warm" else 0
        for run in range(first_run, runs):
            env = os.environ.copy()
            if cache_mode == "off":
                env["CIMBA_CACHE"] = "0"
            else:
                cache = (shared_cache if cache_mode == "warm"
                         else root_path / f"cache-{run}")
                env["CIMBA_CACHE"] = "1"
                env["CIMBA_CACHE_DIR"] = str(cache)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                scenario,
                "--scale",
                str(scale),
                "--workdir",
                str(root_path / ("prime" if run < 0 else f"run-{run}")),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"benchmark worker failed for {scenario}-{scale}:\n"
                    f"{exc.stderr}"
                ) from exc
            if run >= 0:
                samples.append(json.loads(completed.stdout))

    return {
        "scenario": scenario,
        "scale": scale,
        "nodes": samples[0]["nodes"],
        "instances": samples[0]["instances"],
        "runs": runs,
        "cache": cache_mode,
        "phases": {
            phase: _summary([sample["phases"][phase] for sample in samples])
            for phase in PHASES
        },
        "metadata": samples[0]["metadata"],
        "samples": samples,
    }


def _print(results: list[dict[str, Any]]) -> None:
    print("Cimba fresh-process class-callback compilation benchmark")
    print("times are median ± MAD; each sample uses a new interpreter")
    print(
        "scenario             | instances | import | definition | build | "
        "first experiment | cached"
    )
    print(
        "---------------------+-----------+--------+------------+-------+"
        "------------------+-------"
    )
    for result in results:
        label = result["scenario"]
        if label == "synthetic":
            label = f"synthetic-{result['scale']}"
        values = result["phases"]

        def cell(name: str) -> str:
            item = values[name]
            return f"{item['median']:.3f}±{item['mad']:.3f}"

        print(
            f"{label:20} | {result['instances']:9d} | "
            f"{cell('cimba_import'):>6} | {cell('model_definition'):>10} | "
            f"{cell('model_build'):>5} | {cell('first_experiment'):>16} | "
            f"{cell('cached_experiment'):>6}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("assembly-line", "amusement-park", "synthetic"),
        help="scenario to run; defaults to both tutorials and synthetic",
    )
    parser.add_argument("--scales", default="1,10,100,1000")
    parser.add_argument("--cache", choices=("off", "cold", "warm"),
                        default="off")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless models compile and warm-cache runs record hits",
    )
    parser.add_argument("--worker", choices=(
        "assembly-line", "amusement-park", "synthetic"))
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--workdir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.worker is not None:
        if args.workdir is None:
            raise SystemExit("--worker requires --workdir")
        args.workdir.mkdir(parents=True, exist_ok=True)
        print(json.dumps(_worker(args.worker, args.scale, args.workdir)))
        return 0
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    scales = [int(value) for value in args.scales.split(",")]
    if not scales or any(value < 1 for value in scales):
        raise SystemExit("--scales must contain positive integers")
    scenarios = args.scenario or [
        "assembly-line", "amusement-park", "synthetic"]
    cases = [
        (scenario, scale)
        for scenario in scenarios
        for scale in (scales if scenario == "synthetic" else [1])
    ]
    results = [
        _sample(scenario, scale, args.runs, args.cache)
        for scenario, scale in cases
    ]
    if args.check:
        for result in results:
            for sample in result["samples"]:
                precompile = sample["precompile"]
                if precompile is not None and precompile["state"] != "ready":
                    raise SystemExit(
                        f"{result['scenario']} precompile was not ready")
            if args.cache == "warm":
                if not any(
                    sample["precompile"] is not None
                    and sample["precompile"]["cache_hits"] > 0
                    for sample in result["samples"]
                ):
                    raise SystemExit(
                        f"{result['scenario']} recorded no warm-cache hit")
                if not any(
                    sample["callback_cache"]["hits"] > 0
                    for sample in result["samples"]
                ):
                    raise SystemExit(
                        f"{result['scenario']} callbacks recorded no "
                        "warm-cache hit")
    _print(results)
    payload = {
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
