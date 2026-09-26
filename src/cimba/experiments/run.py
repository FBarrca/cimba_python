"""Bind configured objects to per-instance native records and run trials."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from time import monotonic
from types import MappingProxyType
from typing import Any, cast, get_origin

import numpy as np

from cimba.compiler import ensure
from cimba.engine.abi import (
    ENTITY_DESCRIPTOR, HOOK_DESCRIPTOR, INPUT_DESCRIPTOR, INPUT_SLOT,
    PROCESS_DESCRIPTOR, TRIAL_DESCRIPTOR, TRIAL_HEADER,
)
from cimba.engine.runtime import (
    distribution_rows, read_captures, run_blocks, seed_input,
)
from cimba.inputs import dist
from cimba.inputs.sources import DistributionSource, GeneratedRows, Source, TraceSource, trace_rng
from cimba.layout import TrialImageLayout
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


_DISTRIBUTIONS = {
    "exponential": 1, "normal": 2, "gamma": 3, "lognormal": 4,
    "weibull": 5, "poisson": 6, "triangular": 7, "pert": 8,
    "categorical": 9, "uniform": 10, "logistic": 11,
    "cauchy": 12, "erlang": 13, "beta": 14,
    "pert_mod": 15, "rayleigh": 16, "bernoulli": 17,
    "geometric": 18, "binomial": 19, "negative_binomial": 20,
    "pareto": 21, "chi_squared": 22, "f_dist": 23,
    "student_t": 24, "dice": 25,
    "hyperexponential": 26, "hypoexponential": 27,
}
_POLICIES = {"fail": 1, "wrap": 2, "end_trial": 3, "extend": 4}


def _native_name(label: str, limit: int = 31) -> bytes:
    encoded = label.encode("utf-8")
    if len(encoded) <= limit:
        return encoded
    from hashlib import sha256
    suffix = sha256(encoded).hexdigest()[:8].encode()
    return encoded[:limit - 9] + b"-" + suffix


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
        self.layout = TrialImageLayout.of(self.assembly)
        self.seeds = None if seeds is None else np.asarray(seeds, dtype=np.uint64).copy()
        if self.seeds is not None and self.seeds.shape not in {
            (self.replications,), (len(self.design.points), self.replications)
        }:
            raise ExperimentConfigError("seeds must have one per replication or trial")
        self._validate_sources()

    def _validate_sources(self) -> None:
        for point in self.design.points:
            for instance in self.assembly.instances:
                for field in instance.schema.fields:
                    if field.kind not in {"input", "series"}:
                        continue
                    source = point.bindings[instance.model, field.name]
                    if isinstance(source, DistributionSource):
                        try:
                            getattr(dist, source.method)(**dict(source.parameters))
                        except (AttributeError, TypeError, ValueError) as exc:
                            raise ModelDefinitionError(
                                f"{instance.label}.{field.name}: invalid "
                                f"{source.method} source: {exc}") from exc
                    if field.kind == "series" and isfinite(self.window.duration):
                        assert field.step is not None
                        required = max(1, ceil((self.window.warmup +
                                self.window.duration + self.window.cooldown -
                                field.origin) / field.step))
                        if isinstance(source, TraceSource) and (
                            source.on_exhausted == "fail" and
                            len(source.values) < required
                        ):
                            raise ModelDefinitionError(
                                f"{instance.label}.{field.name}: trace has "
                                f"{len(source.values)} rows, needs {required}")

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for instance in self.assembly.instances:
            for field in instance.schema.fields:
                value = instance.values[field.name]
                rows.append({"instance": instance.label, "field": field.name,
                             "kind": field.kind,
                             "source": value.describe() if hasattr(value, "describe") else None})
        return rows

    def _seed(self, point: int, replication: int) -> int:
        if self.seeds is not None:
            return int(self.seeds[replication] if self.seeds.ndim == 1
                       else self.seeds[point, replication])
        return trial_seed(self.seed, replication, point, seeding=self.seeding)

    def _subset(self, seeds: np.ndarray) -> "Experiment":
        chunk = object.__new__(Experiment)
        chunk.model = self.model
        chunk.replications = seeds.shape[1]
        chunk.window = self.window
        chunk.seed = self.seed
        chunk.seeding = self.seeding
        chunk.assembly = self.assembly
        chunk.layout = self.layout
        chunk.design = self.design
        chunk.seeds = seeds
        chunk._swept_children = ()
        return chunk

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
        focused = self._subset(seeds)
        focused.design = Design(self.design.axes, tuple(points))
        return focused.run(workers=workers, on_failure=on_failure,
                           input_memory=input_memory)

    def _descriptors(self, compiled):
        entities = []
        processes = []
        starts = []
        ends = []
        inputs = []
        for instance in self.assembly.instances:
            placement = self.layout.by_object[instance.model]
            record = placement.record
            code = compiled[instance.schema.cls]
            for field in instance.schema.fields:
                offset = record.offset(field.name)
                label = f"{instance.label}.{field.name}"
                if field.kind == "entity":
                    entity = instance.values[field.name]
                    initial = 0
                    if field.value_type is Container:
                        kind, capacity = 1, 1_000_000_000
                        initial = entity.initial
                    elif field.value_type is Resource:
                        kind, capacity = 2, entity.capacity
                    elif field.value_type is Dataset:
                        kind, capacity = 3, 0
                    elif get_origin(field.value_type) is Store:
                        kind, capacity = 4, entity.capacity
                    elif get_origin(field.value_type) is PriorityStore:
                        kind, capacity = 5, entity.capacity
                    elif field.value_type is Condition:
                        kind, capacity = 6, 0
                    else:
                        raise ExperimentConfigError(f"{label}: native handle not implemented")
                    entities.append((placement.offset, offset, kind,
                                     int(entity.captured),
                                     capacity, initial, _native_name(label)))
                elif field.kind in {"input", "series"}:
                    inputs.append((placement.offset, offset,
                                   label.encode("utf-8")[:95]))
            for name, copies, priority in instance.schema.processes:
                processes.append((placement.offset, code.processes[name].address,
                                  priority, copies, _native_name(name)))
        for instance in reversed(self.assembly.instances):
            placement = self.layout.by_object[instance.model]
            code = compiled[instance.schema.cls]
            for name in instance.schema.starts:
                starts.append((placement.offset, code.starts[name].address))
            for name in instance.schema.ends:
                ends.append((placement.offset, code.ends[name].address))
        tables = [
            np.array(entities, dtype=ENTITY_DESCRIPTOR),
            np.array(processes, dtype=PROCESS_DESCRIPTOR),
            np.array(starts, dtype=HOOK_DESCRIPTOR),
            np.array(ends, dtype=HOOK_DESCRIPTOR),
            np.array(inputs, dtype=INPUT_DESCRIPTOR),
        ]
        descriptor = np.zeros(1, dtype=TRIAL_DESCRIPTOR)
        count_fields = ("entity_count", "process_count", "start_count",
                        "end_count", "input_count")
        for key, count_field, table in zip(
            ("entities", "processes", "starts", "ends", "inputs"),
            count_fields, tables,
        ):
            descriptor[key] = table.ctypes.data
            descriptor[count_field] = len(table)
        return tables, descriptor

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
        return self._run_single(workers=workers, on_failure=on_failure,
                                input_memory=input_memory)

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
            part = object.__new__(Experiment)
            part.model = self.model
            part.replications = seeds.shape[1]
            part.window = self.window
            part.seed = self.seed
            part.seeding = self.seeding
            part.assembly = Assembly.of(self.model, picks=picks)
            part.layout = TrialImageLayout.of(part.assembly)
            part.design = Design(self.design.axes, tuple(
                DesignPoint(local, points[position].levels, points[position].bindings)
                for local, position in enumerate(positions)))
            part.seeds = seeds[positions]
            part._swept_children = ()
            parts.append(part._run_single(workers=workers, on_failure=on_failure,
                                          input_memory=input_memory))
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

    def _run_single(self, *, workers: int | None = None,
                    on_failure: str = "record", input_memory: int = 1 << 30) -> Results:
        if on_failure not in {"record", "raise"}:
            raise ExperimentConfigError("on_failure must be record or raise")
        bytes_per_replication = 0
        for point in self.design.points:
            for instance in self.assembly.instances:
                for field in instance.schema.fields:
                    if field.kind not in {"input", "series"}:
                        continue
                    source = point.bindings[instance.model, field.name]
                    if not isinstance(source, GeneratedRows):
                        continue
                    length = source.length_hint or 1024
                    if field.kind == "series" and isfinite(self.window.duration):
                        assert field.step is not None
                        length = max(length, ceil((self.window.warmup +
                                     self.window.duration + self.window.cooldown -
                                     field.origin) / field.step))
                    if source.max_length is not None:
                        length = min(length, source.max_length)
                    bytes_per_replication += length * 8
        if bytes_per_replication * self.replications <= input_memory:
            return self._run_core(workers=workers, on_failure=on_failure,
                                  input_memory=input_memory)
        if bytes_per_replication > input_memory:
            raise ExperimentConfigError(
                f"input_memory={input_memory} cannot hold one replication "
                f"({bytes_per_replication} bytes)")
        chunk_size = max(1, input_memory // bytes_per_replication)
        chunks = []
        for start in range(0, self.replications, chunk_size):
            stop = min(self.replications, start + chunk_size)
            chunk_seeds = np.asarray([
                [self._seed(point, replication)
                 for replication in range(start, stop)]
                for point in range(len(self.design.points))], dtype=np.uint64)
            chunk = self._subset(chunk_seeds)
            chunks.append(chunk._run_core(workers=workers,
                                          on_failure=on_failure,
                                          input_memory=input_memory))
        failed = np.concatenate([chunk.failed for chunk in chunks], axis=1)
        reasons = np.concatenate([chunk.failure_reasons for chunk in chunks], axis=1)
        seeds = np.concatenate([chunk.seeds for chunk in chunks], axis=1)
        outcomes = {}
        for instance in self.assembly.instances:
            fields = {}
            for field in instance.schema.fields:
                records = [chunk[instance.model].__getattr__(field.name)
                           for chunk in chunks
                           if field.kind in {"output", "input", "series"}]
                if field.kind == "output":
                    fields[field.name] = Samples(np.concatenate(
                        [record.values for record in records], axis=1))
                elif field.kind in {"input", "series"}:
                    consumed = np.concatenate(
                        [record.consumed for record in records], axis=1)
                    extended = np.concatenate(
                        [record.extended for record in records], axis=1)
                    def rows(point, replication, records=tuple(records),
                             size=chunk_size):
                        index, local = divmod(replication, size)
                        return records[index].rows(point, local)
                    fields[field.name] = InputRecord(records[0].source,
                                                      consumed, extended, rows)
                elif field.kind == "entity" and instance.values[field.name].captured:
                    fields[field.name] = Signal.join(tuple(
                        chunk[instance.model].__getattr__(field.name)
                        for chunk in chunks))
            outcomes[instance.model] = InstanceResults(fields)
        return Results(MappingProxyType(outcomes), failed, reasons, seeds,
                       RunMeta(MappingProxyType({"workers": workers,
                                         "chunks": len(chunks),
                                         "wall_time": sum(chunk.meta["wall_time"]
                                                          for chunk in chunks),
                                         "compile": MappingProxyType({
                                             "misses": sum(chunk.meta["compile"]["misses"]
                                                           for chunk in chunks),
                                             "wall_time": sum(chunk.meta["compile"]["wall_time"]
                                                              for chunk in chunks)}),
                                         "extensions": sum(chunk.meta["extensions"]
                                                           for chunk in chunks),
                                         "native_failed": sum(chunk.meta["native_failed"]
                                                              for chunk in chunks)})),
                       self.design)

    def _run_core(self, *, workers: int | None = None,
                  on_failure: str = "record", input_memory: int = 1 << 30) -> Results:
        if on_failure not in {"record", "raise"}:
            raise ExperimentConfigError("on_failure must be record or raise")
        if input_memory < 1:
            raise ExperimentConfigError("input_memory must be positive")
        started = monotonic()
        compile_started = monotonic()
        compile_misses_before = ensure.cache_info().misses
        compiled = {instance.schema.cls: ensure(instance.schema)
                    for instance in self.assembly.instances}
        compile_wall_time = monotonic() - compile_started
        compile_misses = ensure.cache_info().misses - compile_misses_before
        tables, descriptor = self._descriptors(compiled)
        points = len(self.design.points)
        reps = self.replications
        count = points * reps
        blocks = np.zeros(count, dtype=f"V{self.layout.size}")
        headers = np.ndarray((count,), dtype=TRIAL_HEADER, buffer=blocks,
                             strides=(self.layout.size,))
        headers["descriptor"] = descriptor.ctypes.data
        headers["warmup"] = self.window.warmup
        headers["duration"] = self.window.duration
        headers["cooldown"] = self.window.cooldown
        seeds = np.empty((points, reps), dtype=np.uint64)
        for p in range(points):
            for r in range(reps):
                seed = self._seed(p, r)
                seeds[p, r] = seed
                headers["seed"][p * reps + r] = seed
                headers["trial_index"][p * reps + r] = p * reps + r

        records = {
            instance.model: np.ndarray(
                (count,), dtype=self.layout.by_object[instance.model].record.dtype,
                buffer=blocks, offset=self.layout.by_object[instance.model].offset,
                strides=(self.layout.size,))
            for instance in self.assembly.instances
        }
        keepalive = [blocks, descriptor, *tables]
        input_rows = {}
        row_groups = {}
        for instance in self.assembly.instances:
            for field in instance.schema.fields:
                if field.kind not in {"input", "series"}:
                    continue
                for point in self.design.points:
                    source = point.bindings[instance.model, field.name]
                    if not isinstance(source, GeneratedRows):
                        continue
                    tag = source.tag or f"{instance.label}.{field.name}"
                    required = 1
                    if field.kind == "series" and isfinite(self.window.duration):
                        assert field.step is not None
                        required = ceil((self.window.warmup + self.window.duration +
                                         self.window.cooldown - field.origin) / field.step)
                    length = max(1, source.length_hint or 1024, required)
                    if source.max_length is not None:
                        length = min(length, source.max_length)
                    key = (id(source), tag, length)
                    group = row_groups.setdefault(key, (source, set()))
                    group[1].update(int(seeds[point.index, r])
                                    for r in range(reps))
        for (source_id, tag, length), (source, source_seeds) in row_groups.items():
            ordered = sorted(source_seeds)
            matrix = source.generate([trace_rng(seed, tag) for seed in ordered],
                                     length)
            keepalive.append(matrix)
            for j, seed in enumerate(ordered):
                input_rows[(source_id, tag, seed, length)] = matrix[j]
        for instance in self.assembly.instances:
            record = records[instance.model]
            placement = self.layout.by_object[instance.model]
            record["class_descriptor"] = compiled[instance.schema.cls].dispatch.ctypes.data
            for field in instance.schema.fields:
                for point in self.design.points:
                    value = point.bindings[instance.model, field.name]
                    for r in range(reps):
                        i = point.index * reps + r
                        if field.kind in {"param", "state", "constant"}:
                            record[field.name][i] = value
                        elif field.kind == "output":
                            record[field.name][i] = (
                                np.nan if field.value_type is float else 0)
                        elif field.kind in {"child", "ref"} and value is not None:
                            record[field.name][i] = (blocks.ctypes.data +
                                i * self.layout.size + self.layout.by_object[value].offset)
                        elif field.kind == "collection":
                            children = cast(list[Model], value)
                            pointers = np.asarray([
                                blocks.ctypes.data + i * self.layout.size +
                                self.layout.by_object[child].offset
                                for child in children], dtype=np.uintp)
                            keepalive.append(pointers)
                            record[field.name][i] = pointers.ctypes.data
                            record[f"{field.name}_length"][i] = len(children)
                        elif field.kind in {"input", "series"}:
                            slot = record[field.name][i]
                            slot["step"] = field.step or 0.0
                            slot["origin"] = field.origin
                            slot["last_bucket"] = -1
                            slot["policy"] = _POLICIES[cast(Source, value).on_exhausted]
                            if isinstance(value, DistributionSource):
                                slot["kind"] = 1
                                slot["distribution"] = _DISTRIBUTIONS[value.method]
                                parameters = dict(value.parameters)
                                if value.method in {"categorical", "hyperexponential",
                                                    "hypoexponential"}:
                                    values = np.asarray(parameters.get("values",
                                                        parameters.get("means")),
                                                        dtype=np.float64)
                                    if value.method == "hypoexponential":
                                        packed = values
                                    else:
                                        weights = np.asarray(parameters["probabilities"],
                                                             dtype=np.float64)
                                        packed = np.concatenate((values, weights))
                                    keepalive.append(packed)
                                    slot["data"] = packed.ctypes.data
                                    slot["length"] = len(values)
                                else:
                                    slot["parameters"][:len(parameters)] = list(parameters.values())
                                from hashlib import sha256
                                label = f"{instance.label}.{field.name}"
                                nonce = int.from_bytes(sha256(label.encode()).digest()[:8], "little")
                                address = blocks.ctypes.data + i * self.layout.size + placement.offset + placement.record.offset(field.name)
                                seed_input(address, int(seeds[point.index, r]) ^ nonce)
                            else:
                                slot["kind"] = 2
                                if isinstance(value, TraceSource):
                                    row = value.values
                                elif isinstance(value, GeneratedRows):
                                    label = value.tag or f"{instance.label}.{field.name}"
                                    required = 1
                                    if field.kind == "series" and isfinite(self.window.duration):
                                        assert field.step is not None
                                        required = ceil((self.window.warmup + self.window.duration +
                                                         self.window.cooldown - field.origin) / field.step)
                                    length = max(1, value.length_hint or 1024,
                                                 required)
                                    if value.max_length is not None:
                                        length = min(length, value.max_length)
                                    key = (id(value), label,
                                           int(seeds[point.index, r]), length)
                                    if key not in input_rows:
                                        input_rows[key] = value.generate(
                                            [trace_rng(int(seeds[point.index, r]), label)],
                                            length)[0]
                                    row = input_rows[key]
                                else:
                                    raise ExperimentConfigError("unsupported input source")
                                keepalive.append(row)
                                slot["data"] = row.ctypes.data
                                slot["length"] = len(row)
        initial_blocks = blocks.copy()
        native_failed = run_blocks(blocks, workers=workers)
        extensions: dict[tuple[int, Model, str], int] = {}
        overrides: dict[int, dict[tuple[Model, str], np.ndarray]] = {}
        for i in range(count):
            point_index, replication = divmod(i, reps)
            while headers["status"][i] == 4:
                exhausted = None
                for instance in self.assembly.instances:
                    record = records[instance.model]
                    for field in instance.schema.fields:
                        if field.kind not in {"input", "series"}:
                            continue
                        if record[field.name][i]["status"] == 3:
                            exhausted = (instance, field)
                            break
                    if exhausted is not None:
                        break
                if exhausted is None:
                    # A spawned model can exhaust a clone of a static row
                    # source. Its slot is not in the static instance graph,
                    # so extend every generated row bound in this design
                    # point; unchanged prefixes preserve the other inputs.
                    targets = [
                        (instance, field)
                        for instance in self.assembly.instances
                        for field in instance.schema.fields
                        if field.kind in {"input", "series"} and
                        isinstance(self.design.points[point_index].bindings[
                            instance.model, field.name], GeneratedRows)
                    ]
                else:
                    targets = [exhausted]
                if not targets:
                    headers["status"][i] = 3
                    headers["error"][i] = b"input extension requested without a row source"
                    break
                extension_failed = False
                for instance, field in targets:
                    source = self.design.points[point_index].bindings[
                        instance.model, field.name]
                    key = (i, instance.model, field.name)
                    count_extended = extensions.get(key, 0)
                    if not isinstance(source, GeneratedRows) or (
                        not source.prefix_stable or count_extended >= 6
                    ):
                        headers["status"][i] = 3
                        headers["error"][i] = (
                            f"{instance.label}.{field.name}: extension limit reached"
                        ).encode()
                        extension_failed = True
                        break
                    old_slot = records[instance.model][field.name][i]
                    old_length = int(old_slot["length"])
                    new_length = old_length * 2
                    if source.max_length is not None and new_length > source.max_length:
                        headers["status"][i] = 3
                        headers["error"][i] = (
                            f"{instance.label}.{field.name}: source maximum length reached"
                        ).encode()
                        extension_failed = True
                        break
                    tag = source.tag or f"{instance.label}.{field.name}"
                    row = source.generate(
                        [trace_rng(int(seeds[point_index, replication]), tag)],
                        new_length)[0]
                    old = overrides.get(i, {}).get((instance.model, field.name))
                    if old is None:
                        old = input_rows.get((id(source), tag,
                                              int(seeds[point_index, replication]),
                                              old_length))
                    if old is not None and not np.array_equal(row[:old_length], old):
                        raise RuntimeError(f"{instance.label}.{field.name}: source is not prefix-stable")
                    overrides.setdefault(i, {})[(instance.model, field.name)] = row
                    extensions[key] = count_extended + 1
                    keepalive.append(row)
                if extension_failed:
                    break
                blocks[i] = initial_blocks[i]
                for (model, name), bound_row in overrides[i].items():
                    slot = records[model][name][i]
                    slot["data"] = bound_row.ctypes.data
                    slot["length"] = len(bound_row)
                native_failed += run_blocks(blocks[i:i + 1], workers=1)
        statuses = headers["status"].reshape(points, reps).copy()
        failed = statuses != 2
        reasons = headers["error"].reshape(points, reps).astype(str).copy()
        captured_rows = read_captures(headers, len(tables[0]))
        if on_failure == "raise" and failed.any():
            raise TrialsFailed(f"{int(failed.sum())} trials failed: {reasons[failed][0]}")
        entity_positions = {}
        for instance in self.assembly.instances:
            for field in instance.schema.fields:
                if field.kind == "entity":
                    entity_positions[instance.model, field.name] = len(entity_positions)
        outcomes = {}
        for instance in self.assembly.instances:
            fields = {}
            record = records[instance.model]
            for field in instance.schema.fields:
                if field.kind == "output":
                    values = np.asarray(record[field.name],
                                        dtype=np.float64).reshape(points, reps).copy()
                    values[failed] = np.nan
                    fields[field.name] = Samples(values)
                elif field.kind in {"input", "series"}:
                    source = tuple(point.bindings[instance.model, field.name].describe()
                                   for point in self.design.points)
                    consumed = np.asarray(record[field.name]["cursor"]).reshape(points, reps).copy()
                    extended = np.zeros((points, reps), dtype=np.int64)
                    for p in range(points):
                        for r in range(reps):
                            extended[p, r] = extensions.get(
                                (p * reps + r, instance.model, field.name), 0)
                    def rows(p, r, model=instance.model, name=field.name,
                             consumed_for_field=consumed):
                        chosen = self.design.points[p].bindings[model, name]
                        if isinstance(chosen, TraceSource):
                            return chosen.values
                        if isinstance(chosen, GeneratedRows):
                            label = chosen.tag or f"{self.assembly.by_object[model].label}.{name}"
                            return chosen.generate([trace_rng(int(seeds[p, r]), label)],
                                                   max(1, int(consumed_for_field[p, r])))[0][
                                                       :int(consumed_for_field[p, r])]
                        if isinstance(chosen, DistributionSource):
                            from hashlib import sha256
                            label = f"{self.assembly.by_object[model].label}.{name}"
                            nonce = int.from_bytes(sha256(label.encode()).digest()[:8], "little")
                            parameters = dict(chosen.parameters)
                            categorical = None
                            if chosen.method == "categorical":
                                categorical = (parameters["values"],
                                               parameters["probabilities"])
                                numeric = ()
                            elif chosen.method == "hyperexponential":
                                categorical = (parameters["means"],
                                               parameters["probabilities"])
                                numeric = ()
                            elif chosen.method == "hypoexponential":
                                categorical = (parameters["means"], ())
                                numeric = ()
                            else:
                                numeric = tuple(parameters.values())
                            return distribution_rows(
                                _DISTRIBUTIONS[chosen.method], numeric,
                                int(seeds[p, r]) ^ nonce,
                                int(consumed_for_field[p, r]),
                                categorical)
                        raise ValueError("unsupported source")
                    fields[field.name] = InputRecord(source, consumed, extended, rows)
                elif field.kind == "entity" and instance.values[field.name].captured:
                    position = entity_positions[instance.model, field.name]
                    empty = lambda: (np.empty(0, dtype=np.float64),
                                     np.empty(0, dtype=np.float64))
                    trials = tuple(tuple(
                        captured_rows[p * reps + r][position]
                        if captured_rows[p * reps + r] is not None else empty()
                        for r in range(reps)) for p in range(points))
                    fields[field.name] = Signal(trials)
            outcomes[instance.model] = InstanceResults(fields)
        return Results(MappingProxyType(outcomes), failed, reasons, seeds,
                       RunMeta(MappingProxyType({"workers": workers,
                                         "wall_time": monotonic() - started,
                                         "compile": MappingProxyType({
                                             "misses": compile_misses,
                                             "wall_time": compile_wall_time}),
                                         "extensions": sum(extensions.values()),
                                         "native_failed": native_failed})),
                       self.design)
