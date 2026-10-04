"""Bind configured objects to per-instance native records and run trials."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any

import numpy as np

from cimba.inputs.sources import Source
from cimba.modeling import Condition, Container, Dataset, Model, PriorityStore, Resource, Store, Sweep
from cimba.results import InputRecord, InstanceResults, Results, RunMeta, Samples, Signal
from cimba.schema import Assembly, ModelDefinitionError
from .design import Design, DesignPoint, trial_seed


class ExperimentConfigError(ValueError):
    pass


class TrialsFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class Window:
    warmup: float = 0.0
    duration: float = float("inf")
    cooldown: float = 0.0

    def __post_init__(self):
        if self.warmup < 0 or self.cooldown < 0 or self.duration <= 0:
            raise ExperimentConfigError("window times must be nonnegative with positive duration")
        if not isfinite(self.duration) and (self.warmup or self.cooldown):
            raise ExperimentConfigError("until-idle window cannot have warmup or cooldown")

    @classmethod
    def until_idle(cls) -> "Window":
        return cls()


class Experiment:
    def __init__(self, model: Model, *, replications: int = 1,
                 window: Window | None = None, seed: int = 0,
                 seeding: str = "common", seeds: np.ndarray | None = None):
        if replications < 1:
            raise ExperimentConfigError("replications must be positive")
        if seeding not in {"common", "independent"}:
            raise ExperimentConfigError("seeding must be common or independent")
        self.model = model
        self.replications = int(replications)
        self.window = window or Window()
        self.seed = int(seed)
        self.seeding = seeding
        # The full tree includes every option of a swept child model; it is used
        # for validation and design expansion. Trials run on per-option trees.
        self.assembly = Assembly.of(model)
        self.design = Design.of(self.assembly)
        self._swept_children = tuple(
            (instance.model, field.name)
            for instance in self.assembly.instances for field in instance.schema.fields
            if field.kind == "child" and isinstance(instance.values[field.name], Sweep))
        # With swept children this lays out every option; runs then use one
        # layout per option (see _run_variants).
        self.seeds = None if seeds is None else np.asarray(seeds, dtype=np.uint64).copy()
        if self.seeds is not None and self.seeds.shape not in {
            (self.replications,), (len(self.design.points), self.replications)
        }:
            raise ExperimentConfigError("seeds must have one per replication or trial")
        from .snapshot import Snapshot
        self.snapshot = Snapshot(self.assembly, self.window)
        self.layout = self.snapshot.layout
        self.snapshot.validate(self.design)

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for instance in self.assembly.instances:
            for field in instance.schema.fields:
                value = instance.values[field.name]
                rows.append({"instance": instance.label, "field": field.name,
                             "kind": field.kind,
                             "source": value.describe() if isinstance(value, Source) else None})
        return rows

    def _seed(self, point: int, replication: int) -> int:
        if self.seeds is not None:
            return int(self.seeds[replication] if self.seeds.ndim == 1
                       else self.seeds[point, replication])
        return trial_seed(self.seed, replication, point, seeding=self.seeding)

    def only(self, *, trials, workers: int | None = 1,
             on_failure: str = "record", input_memory: int = 1 << 30) -> Results:
        """Rerun selected flattened trial indices with their original seeds."""
        selected = tuple(int(index) for index in trials)
        total = len(self.design.points) * self.replications
        if not selected or any(index < 0 or index >= total for index in selected):
            raise ExperimentConfigError(f"trial indices must be in [0, {total})")
        seeds = np.empty((len(selected), 1), dtype=np.uint64)
        points = []
        for new_index, trial_index in enumerate(selected):
            point_index, replication = divmod(trial_index, self.replications)
            original = self.design.points[point_index]
            points.append(DesignPoint(new_index, original.levels,
                                      original.bindings))
            seeds[new_index, 0] = self._seed(point_index, replication)
        if self._swept_children:
            return self._run_variants(points, seeds, workers=workers,
                                      on_failure=on_failure, input_memory=input_memory)
        return self.snapshot.run(Design(self.design.axes, tuple(points)), seeds,
                                 workers=workers, on_failure=on_failure,
                                 input_memory=input_memory)

    def run(self, *, workers: int | None = None,
            on_failure: str = "record", input_memory: int = 1 << 30) -> Results:
        """Run every trial; returns immutable :class:`Results`."""
        if input_memory < 1:
            raise ExperimentConfigError("input_memory must be positive")
        if self._swept_children:
            seeds = np.asarray([[self._seed(point.index, r) for r in range(self.replications)]
                                for point in self.design.points], dtype=np.uint64)
            return self._run_variants(list(self.design.points), seeds, workers=workers,
                                      on_failure=on_failure, input_memory=input_memory)
        if self.seeds is not None:
            seeds = np.broadcast_to(self.seeds, (len(self.design.points), self.replications)).copy()
        elif self.seeding == 'common':
            common = [trial_seed(self.seed, r) for r in range(self.replications)]
            seeds = np.broadcast_to(np.asarray(common, dtype=np.uint64),
                                    (len(self.design.points), self.replications)).copy()
        else:
            seeds = np.asarray([[self._seed(p, r) for r in range(self.replications)]
                                for p in range(len(self.design.points))], dtype=np.uint64)
        return self.snapshot.run(self.design, seeds, workers=workers,
                                 on_failure=on_failure, input_memory=input_memory)

    def _run_variants(self, points: list[DesignPoint], seeds: np.ndarray, *,
                      workers: int | None, on_failure: str, input_memory: int) -> Results:
        """Run each combination of swept child models on its own model tree.

        A trial contains only the options its design point selects: the other
        options do not exist in it, so their processes and hooks never run.
        Each group reuses the original trial seeds, keeping common random
        numbers across options, and the parts are merged into one Results.
        """
        groups: dict[tuple[int, ...], list[int]] = {}
        for position, point in enumerate(points):
            key = tuple(id(point.bindings[swept]) for swept in self._swept_children)
            groups.setdefault(key, []).append(position)
        parts = []
        for positions in groups.values():
            first = points[positions[0]]
            picks = {swept: first.bindings[swept] for swept in self._swept_children}
            from .snapshot import Snapshot
            snapshot = Snapshot(Assembly.of(self.model, picks=picks), self.window)
            design = Design(self.design.axes, tuple(
                DesignPoint(local, points[position].levels, points[position].bindings)
                for local, position in enumerate(positions)))
            parts.append(snapshot.run(design, seeds[positions], workers=workers,
                                      on_failure=on_failure, input_memory=input_memory))
        return self._merge(points, list(groups.values()), parts, seeds, workers)

    def _merge(self, points, groups, parts, seeds, workers) -> Results:
        count, replications = seeds.shape
        failed = np.ones((count, replications), dtype=bool)
        reasons = np.full((count, replications), "", dtype=object)
        for positions, part in zip(groups, parts):
            failed[positions] = part.failed
            reasons[positions] = part.failure_reasons
        outcomes = {}
        for instance in self.assembly.instances:
            model = instance.model
            present = [(positions, part) for positions, part in zip(groups, parts)
                       if model in part.by_object]
            fields = {}
            if not present:
                # An option that no selected trial uses: report it as absent.
                for field in instance.schema.fields:
                    if field.kind == "output":
                        fields[field.name] = Samples(np.full((count, replications), np.nan))
                    elif field.kind in {"input", "series"}:
                        def absent(point, replication, label=instance.label):
                            raise KeyError(f"{label} is not part of design point {point}")
                        fields[field.name] = InputRecord(
                            (None,) * count, np.zeros((count, replications), dtype=np.uint64),
                            np.zeros((count, replications), dtype=np.int64), absent)
                    elif field.kind == "entity" and instance.values[field.name].captured:
                        fields[field.name] = Signal(tuple(
                            tuple((np.empty(0), np.empty(0)) for _ in range(replications))
                            for _ in range(count)))
                outcomes[model] = InstanceResults(fields)
                continue
            for name, value in present[0][1][model]._fields.items():
                if isinstance(value, Samples):
                    merged = np.full((count, replications), np.nan)
                    for positions, part in present:
                        merged[positions] = part[model]._fields[name].values
                    fields[name] = Samples(merged)
                elif isinstance(value, InputRecord):
                    source: list[Any] = [None] * count
                    consumed = np.zeros((count, replications), dtype=value.consumed.dtype)
                    extended = np.zeros((count, replications), dtype=value.extended.dtype)
                    owners = {}
                    for positions, part in present:
                        record = part[model]._fields[name]
                        consumed[positions] = record.consumed
                        extended[positions] = record.extended
                        for local, position in enumerate(positions):
                            source[position] = record.source[local]
                            owners[position] = (record, local)

                    def rows(point, replication, owners=owners, label=instance.label):
                        if point not in owners:
                            raise KeyError(f"{label} is not part of design point {point}")
                        record, local = owners[point]
                        return record.rows(local, replication)
                    fields[name] = InputRecord(tuple(source), consumed, extended, rows)
                elif isinstance(value, Signal):
                    trials: list[Any] = [tuple((np.empty(0), np.empty(0))
                                               for _ in range(replications))
                                         for _ in range(count)]
                    for positions, part in present:
                        signal = part[model]._fields[name]
                        for local, position in enumerate(positions):
                            trials[position] = signal.trials[local]
                    fields[name] = Signal(tuple(trials))
            outcomes[model] = InstanceResults(fields)
        meta = {
            "workers": workers,
            "wall_time": sum(part.meta["wall_time"] for part in parts),
            "compile": MappingProxyType({
                "misses": sum(part.meta["compile"]["misses"] for part in parts),
                "wall_time": sum(part.meta["compile"]["wall_time"] for part in parts)}),
            "extensions": sum(part.meta["extensions"] for part in parts),
            "native_failed": sum(part.meta["native_failed"] for part in parts),
            "variants": len(parts),
        }
        return Results(MappingProxyType(outcomes), failed, reasons.astype(str), seeds,
                       RunMeta(MappingProxyType(meta)),
                       Design(self.design.axes, tuple(points)))
