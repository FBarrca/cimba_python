"""Bind configured objects to per-instance native records and run trials."""

from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from math import ceil, isfinite
from time import monotonic, process_time
from types import MappingProxyType
from typing import Any, cast, get_origin

import numpy as np

from cimba.compiler import ensure
from cimba.engine.abi import (
    ENTITY_DESCRIPTOR, HOOK_DESCRIPTOR, INPUT_DESCRIPTOR, INPUT_EMPTY, INPUT_SLOT,
    PROCESS_DESCRIPTOR, TRIAL_DESCRIPTOR, TRIAL_HEADER,
)
from cimba.engine.runtime import (
    distribution_rows, read_captures, run_blocks, seed_inputs,
)
from cimba.inputs import dist
from cimba.inputs.sources import (
    DistributionSource, EmptySource, GeneratedRows, Source, TraceSource, trace_rng,
)
from cimba.layout import TrialImageLayout
from cimba.modeling import Condition, Container, Dataset, Model, PriorityStore, Resource, Store, Sweep
from cimba.results import InputRecord, InstanceResults, Results, RunMeta, Samples, Signal
from cimba.schema import Assembly, ModelDefinitionError, scalar_value
from .design import Design
from .run import ExperimentConfigError, TrialsFailed, Window


_DISTRIBUTIONS = {
    "exponential": 1, "normal": 2, "gamma": 3, "lognormal": 4,
    "weibull": 5, "poisson": 6, "triangular": 7, "pert": 8,
    "categorical": 9, "uniform": 10, "logistic": 11,
    "cauchy": 12, "erlang": 13, "beta": 14, "pert_mod": 15,
    "rayleigh": 16, "bernoulli": 17, "geometric": 18, "binomial": 19,
    "negative_binomial": 20, "pareto": 21, "chi_squared": 22,
    "f_dist": 23, "student_t": 24, "dice": 25,
    "hyperexponential": 26, "hypoexponential": 27,
}
_POLICIES = {"fail": 1, "wrap": 2, "end_trial": 3, "extend": 4}


def _native_name(label: str, limit: int = 31) -> bytes:
    encoded = label.encode("utf-8")
    if len(encoded) <= limit:
        return encoded
    suffix = sha256(encoded).hexdigest()[:8].encode()
    return encoded[:limit - 9] + b"-" + suffix



class Snapshot:
    """Reusable assembly, layout, compiled callbacks and descriptor tables.

    Batches supply explicit points and trial seeds. No live model values are
    read after construction; captures can be disabled without changing the tree.
    """

    def __init__(self, assembly: Assembly, window: Window):
        self.assembly = assembly
        self.window = window
        self.layout = TrialImageLayout.of(assembly)
        self._compiled = None
        self._tables = {}
        self._row_cache = OrderedDict()
        self._row_bytes = 0
        self._validated_sources = {}

    def prepare(self, captures: bool):
        started = monotonic()
        misses = ensure.cache_info().misses
        if self._compiled is None:
            self._compiled = {instance.schema.cls: ensure(instance.schema)
                              for instance in self.assembly.instances}
            self._tables = {True: self._descriptors(self._compiled)}
        if captures not in self._tables:
            tables, descriptor = self._tables[True]
            tables = [table.copy() for table in tables]
            tables[0]['reserved'] = 0
            descriptor = descriptor.copy()
            for name, table in zip(('entities', 'processes', 'starts', 'ends', 'inputs'), tables):
                descriptor[name] = table.ctypes.data
            self._tables[captures] = tables, descriptor
        return (self._compiled, *self._tables[captures],
                ensure.cache_info().misses - misses, monotonic() - started)

    def rows(self, source, tag, seeds, length, memory):
        """Generate missing rows together, retaining only rows within the limit."""
        self.trim_rows(memory)
        rows = {}
        missing = []
        for seed in sorted(set(seeds)):
            key = (id(source), tag, seed, length)
            if key in self._row_cache:
                rows[key] = self._row_cache[key]
                self._row_cache.move_to_end(key)
            else:
                missing.append(seed)
        if missing:
            matrix = source.generate([trace_rng(seed, tag) for seed in missing], length)
            for i, seed in enumerate(missing):
                key = (id(source), tag, seed, length)
                row = matrix[i].copy()
                row.flags.writeable = False
                rows[key] = row
                self._row_cache[key] = row
                self._row_bytes += row.nbytes
                self.trim_rows(memory)
        return rows

    def trim_rows(self, memory):
        while self._row_bytes > memory and self._row_cache:
            _, row = self._row_cache.popitem(last=False)
            self._row_bytes -= row.nbytes

    def run(self, design: Design, seeds: np.ndarray, *, workers=None,
            on_failure='record', input_memory=1 << 30, captures=True) -> Results:
        if input_memory < 1:
            raise ExperimentConfigError('input_memory must be positive')
        seeds = np.asarray(seeds, dtype=np.uint64)
        if seeds.ndim != 2 or seeds.shape[0] != len(design.points) or not seeds.shape[1]:
            raise ExperimentConfigError('seeds must have shape (points, replications)')
        if not design.points:
            raise ExperimentConfigError('a batch needs at least one point')
        if any(point.index != i for i, point in enumerate(design.points)):
            raise ExperimentConfigError('batch point indices must be consecutive')
        self.validate(design)
        return Batch(self, design, seeds, captures)._run_single(
            workers=workers, on_failure=on_failure, input_memory=input_memory)

    def validate(self, design: Design) -> None:
        for point in design.points:
            for instance in self.assembly.instances:
                for field in instance.schema.fields:
                    if field.kind not in {"input", "series"}:
                        continue
                    source = point.bindings[instance.model, field.name]
                    key = (id(source), field.kind, field.step, field.origin)
                    if key in self._validated_sources:
                        continue
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
                    self._validated_sources[key] = source

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
                processes.append((placement.offset, code.entries[name],
                                  priority, copies, _native_name(name)))
        for instance in reversed(self.assembly.instances):
            placement = self.layout.by_object[instance.model]
            code = compiled[instance.schema.cls]
            for name in instance.schema.starts:
                starts.append((placement.offset, code.entries[name]))
            for name in instance.schema.ends:
                ends.append((placement.offset, code.entries[name]))
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


class Batch:
    """Bindings and seeds of one execution against a reusable snapshot."""

    def __init__(self, snapshot, design, seeds, captures):
        self.snapshot = snapshot
        self.assembly = snapshot.assembly
        self.layout = snapshot.layout
        self.window = snapshot.window
        self.design = design
        self.seeds = seeds
        self.replications = seeds.shape[1]
        self.captures = captures

    def _run_single(self, *, workers: int | None = None,
                    on_failure: str = "record", input_memory: int = 1 << 30) -> Results:
        if on_failure not in {"record", "raise"}:
            raise ExperimentConfigError("on_failure must be record or raise")
        bytes_per_replication = 0
        row_sizes = {}
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
                    tag = source.tag or f'{instance.label}.{field.name}'
                    row_sizes[id(source), tag, length] = length * 8
        common = np.all(self.seeds == self.seeds[0])
        bytes_per_replication = sum(row_sizes.values()) if common else sum(
            size * len(self.design.points) for size in row_sizes.values())
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
                [self.seeds[point, replication]
                 for replication in range(start, stop)]
                for point in range(len(self.design.points))], dtype=np.uint64)
            chunk = Batch(self.snapshot, self.design, chunk_seeds, self.captures)
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
                elif field.kind == "entity" and self.captures and instance.values[field.name].captured:
                    fields[field.name] = Signal.join(tuple(
                        chunk[instance.model].__getattr__(field.name)
                        for chunk in chunks))
            outcomes[instance.model] = InstanceResults(fields)
        return Results(MappingProxyType(outcomes), failed, reasons, seeds,
                       RunMeta(MappingProxyType({"workers": workers,
                                         "chunks": len(chunks),
                                         "native_batches": tuple(batch for chunk in chunks
                                                                 for batch in chunk.meta.native_batches),
                                         **{name: sum(chunk.meta[name] for chunk in chunks)
                                            for name in ('native_wall_time', 'native_cpu_time', 'native_calls', 'host_wall_time')},
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
        compiled, tables, descriptor, compile_misses, compile_wall_time = self.snapshot.prepare(self.captures)
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
        seeds = self.seeds.copy()
        headers['seed'] = seeds.ravel()
        headers['trial_index'] = np.arange(count)

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
            input_rows.update(self.snapshot.rows(source, tag, source_seeds, length, input_memory))
        for instance in self.assembly.instances:
            record = records[instance.model]
            placement = self.layout.by_object[instance.model]
            record['class_descriptor'] = compiled[instance.schema.cls].dispatch.ctypes.data
            for field in instance.schema.fields:
                target = record[field.name]
                if field.kind == 'output':
                    target[:] = np.nan if field.value_type is float else 0
                    continue
                for point in self.design.points:
                    value = point.bindings[instance.model, field.name]
                    start = point.index * reps
                    stop = start + reps
                    indices = np.arange(start, stop, dtype=np.uintp)
                    if field.kind in {'param', 'state', 'constant'}:
                        target[start:stop] = scalar_value(field, value, f'{instance.label}.{field.name}')
                    elif field.kind in {'child', 'ref'} and value is not None:
                        target[start:stop] = blocks.ctypes.data + indices * self.layout.size + self.layout.by_object[value].offset
                    elif field.kind == 'collection':
                        children = cast(list[Model], value)
                        offsets = np.asarray([self.layout.by_object[child].offset if child is not None else 0
                                              for child in children], dtype=np.uintp)
                        pointers = blocks.ctypes.data + indices[:, None] * self.layout.size + offsets
                        for j, child in enumerate(children):
                            if child is None:
                                pointers[:, j] = 0
                        keepalive.append(pointers)
                        target[start:stop] = pointers.ctypes.data + np.arange(reps, dtype=np.uintp) * len(children) * pointers.dtype.itemsize
                        record[f'{field.name}_length'][start:stop] = len(children)
                    elif field.kind in {'input', 'series'}:
                        slots = target[start:stop]
                        slots['step'] = field.step or 0.0
                        slots['origin'] = field.origin
                        slots['last_bucket'] = -1
                        slots['policy'] = _POLICIES[cast(Source, value).on_exhausted]
                        if isinstance(value, EmptySource):
                            slots['kind'] = INPUT_EMPTY
                        elif isinstance(value, DistributionSource):
                            slots['kind'] = 1
                            slots['distribution'] = _DISTRIBUTIONS[value.method]
                            parameters = dict(value.parameters)
                            if value.method in {'categorical', 'hyperexponential', 'hypoexponential'}:
                                values = np.asarray(parameters.get('values', parameters.get('means')), dtype=np.float64)
                                packed = values if value.method == 'hypoexponential' else np.concatenate((values, parameters['probabilities']))
                                keepalive.append(packed)
                                slots['data'] = packed.ctypes.data
                                slots['length'] = len(values)
                            else:
                                slots['parameters'][:, :len(parameters)] = list(parameters.values())
                            label = value.tag if value.tag is not None else f'{instance.label}.{field.name}'
                            nonce = int.from_bytes(sha256(label.encode()).digest()[:8], 'little')
                            seed_inputs(slots, seeds[point.index] ^ np.uint64(nonce))
                        else:
                            slots['kind'] = 2
                            if isinstance(value, TraceSource):
                                slots['data'] = value.values.ctypes.data
                                slots['length'] = len(value.values)
                            elif isinstance(value, GeneratedRows):
                                tag = value.tag or f'{instance.label}.{field.name}'
                                required = 1
                                if field.kind == 'series' and isfinite(self.window.duration):
                                    required = ceil((self.window.warmup + self.window.duration + self.window.cooldown - field.origin) / field.step)
                                length = max(1, value.length_hint or 1024, required)
                                if value.max_length is not None:
                                    length = min(length, value.max_length)
                                bound_rows = [input_rows[(id(value), tag, int(seed), length)] for seed in seeds[point.index]]
                                slots['data'] = [row.ctypes.data for row in bound_rows]
                                slots['length'] = length
                            else:
                                raise ExperimentConfigError('unsupported input source')
        extendable = any(isinstance(source, GeneratedRows) and source.on_exhausted == 'extend'
                         for point in self.design.points for source in point.bindings.values())
        initial_blocks = blocks.copy() if extendable else None
        native_started, cpu_started = monotonic(), process_time()
        native_failed = run_blocks(blocks, workers=workers)
        native_wall = monotonic() - native_started
        native_cpu = process_time() - cpu_started
        native_calls = 1
        native_batches = [{"trials": count, "candidates": points,
                           "native_wall_time": native_wall, "native_cpu_time": native_cpu}]
        extensions: dict[tuple[int, Model, str], int] = {}
        overrides: dict[int, dict[tuple[Model, str], np.ndarray]] = {}
        for index in np.flatnonzero(headers['status'] == 4):
            i = int(index)
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
                    seed = int(seeds[point_index, replication])
                    row = self.snapshot.rows(source, tag, [seed], new_length, input_memory)[
                        id(source), tag, seed, new_length]
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
                assert initial_blocks is not None
                blocks[i] = initial_blocks[i]
                for (model, name), bound_row in overrides[i].items():
                    slot = records[model][name][i]
                    slot["data"] = bound_row.ctypes.data
                    slot["length"] = len(bound_row)
                native_started, cpu_started = monotonic(), process_time()
                native_failed += run_blocks(blocks[i:i + 1], workers=1)
                elapsed = monotonic() - native_started
                cpu = process_time() - cpu_started
                native_wall += elapsed
                native_cpu += cpu
                native_calls += 1
                native_batches.append({"trials": 1, "candidates": 1,
                                       "native_wall_time": elapsed, "native_cpu_time": cpu})
        statuses = headers["status"].reshape(points, reps).copy()
        failed = statuses != 2
        reasons = (headers["error"].reshape(points, reps).astype(str)
                   if failed.any() else np.full((points, reps), '', dtype='<U1'))
        captured_rows = read_captures(headers, len(tables[0])) if tables[0]['reserved'].any() else None
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
                    for (i, model, name), times in extensions.items():
                        if model is instance.model and name == field.name:
                            extended.flat[i] = times
                    def rows(p, r, model=instance.model, name=field.name,
                             consumed_for_field=consumed):
                        chosen = self.design.points[p].bindings[model, name]
                        if isinstance(chosen, EmptySource):
                            return np.empty(0, dtype=np.float64)
                        if isinstance(chosen, TraceSource):
                            return chosen.values
                        if isinstance(chosen, GeneratedRows):
                            label = chosen.tag or f"{self.assembly.by_object[model].label}.{name}"
                            return chosen.generate([trace_rng(int(seeds[p, r]), label)],
                                                   max(1, int(consumed_for_field[p, r])))[0][
                                                       :int(consumed_for_field[p, r])]
                        if isinstance(chosen, DistributionSource):
                            from hashlib import sha256
                            label = (chosen.tag if chosen.tag is not None else
                                     f"{self.assembly.by_object[model].label}.{name}")
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
                elif field.kind == "entity" and self.captures and instance.values[field.name].captured:
                    assert captured_rows is not None
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
                                         "native_wall_time": native_wall,
                                         "native_cpu_time": native_cpu,
                                         "native_calls": native_calls,
                                         "native_batches": tuple(native_batches),
                                         "host_wall_time": monotonic() - started - native_wall,
                                         "extensions": sum(extensions.values()),
                                         "native_failed": native_failed})),
                       self.design)
