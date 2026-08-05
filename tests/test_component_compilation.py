"""Benchmark compilation of representative Cimba component models.

Run with ``uv run pytest tests/test_component_compilation.py -s -q`` to see
the measurements.  This is intentionally a reporting benchmark rather than
a threshold-based performance test: compiler and CPU differences make a fixed
limit unsuitable for the test suite.
"""

import time
from collections.abc import Callable
from pathlib import Path

import cimba.sim as sim
from tutorial import tut_3_1, tut_5_1


def _build_amusement_park() -> sim.Model:
    """Create a fresh park and include its aggregate collection callback."""

    model = tut_3_1.Park()
    model.collect(tut_3_1.park_stats)
    return model


def _representative_models(tmp_path: Path) -> tuple[
    tuple[str, Callable[[], sim.Model]], ...
]:
    return (
        (
            "assembly-line",
            lambda: tut_5_1.build_model(tmp_path / "assembly-line"),
        ),
        (
            "amusement-park",
            _build_amusement_park,
        ),
    )


def _component_counts(model: sim.Model) -> tuple[int, int]:
    """Return structural component nodes and concrete component instances."""

    roots = model._component_roots.values()
    nodes = tuple(node for root in roots for node in root.walk())
    return len(nodes), sum(len(node.instances) for node in nodes)


def test_component_compilation_benchmark(tmp_path):
    """Measure cold compilation for realistic component-heavy models."""

    measurements: list[tuple[str, int, int, float, float, float]] = []
    for name, build_model in _representative_models(tmp_path):
        started = time.perf_counter()
        model = build_model()
        build_seconds = time.perf_counter() - started

        started = time.perf_counter()
        experiment = model.experiment(
            replications=1,
            duration=1.0,
            warmup=0.0,
            cooldown=0.0,
        )
        compile_seconds = time.perf_counter() - started

        started = time.perf_counter()
        cached_experiment = model.experiment(
            replications=1,
            duration=1.0,
            warmup=0.0,
            cooldown=0.0,
        )
        cached_seconds = time.perf_counter() - started

        assert model._compiled is not None
        assert experiment.trials.dtype == cached_experiment.trials.dtype
        component_nodes, component_instances = _component_counts(model)
        measurements.append(
            (
                name,
                component_nodes,
                component_instances,
                build_seconds,
                compile_seconds,
                cached_seconds,
            )
        )

    print("\nCimba representative component compilation benchmark")
    print("model           | nodes | instances | model build | cold compile | cached experiment")
    print("----------------+-------+-----------+-------------+--------------+------------------")
    for name, component_nodes, component_instances, build_seconds, compile_seconds, cached_seconds in measurements:
        print(
            f"{name:15} | {component_nodes:5d} | {component_instances:9d} | "
            f"{build_seconds:10.4f}s | {compile_seconds:11.4f}s | "
            f"{cached_seconds:15.4f}s"
        )
