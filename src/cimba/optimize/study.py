"""Sample-average optimization over reusable Cimba trial batches."""

from __future__ import annotations

from math import isfinite
from numbers import Integral
from os import cpu_count
from time import monotonic
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import scipy

from cimba.analysis import compare, summary
from cimba.experiments import Experiment, ExperimentConfigError, Snapshot, TrialsFailed, Window, trial_seed
from cimba.inputs.sources import trace_rng
from cimba.modeling import Decision, Model
from cimba.results import RunMeta
from cimba.schema import Assembly

from .de import DifferentialEvolution, SearchStopped
from .outcome import (BatchRow, Estimate, Evaluation, Finalist, HistoryRow,
                      OptimizationError, Optimum, Progress, Table)
from .space import Space


def _positive_integer(value, name, minimum=1):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ExperimentConfigError(f"{name} must be an integer >= {minimum}")
    return int(value)


class Optimization:
    def __init__(self, model, *, minimize=None, maximize=None, replications=32,
                 window=None, seed=0):
        if (minimize is None) == (maximize is None):
            raise ExperimentConfigError("pass exactly one of minimize or maximize")
        objective = minimize if minimize is not None else maximize
        if not callable(objective):
            raise TypeError("objective must be callable")
        self.objective = objective
        self.direction = 1 if minimize is not None else -1
        self.model = model
        self.replications = _positive_integer(replications, 'replications')
        self.window = window or Window()
        if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
            raise ExperimentConfigError("seed must be a nonnegative integer")
        self.seed = int(seed)
        self.assembly = Assembly.of(model)
        self.space = Space(self.assembly)
        self.snapshot = Snapshot(self.assembly, self.window)
        self.snapshot.validate(self.space.design([tuple(d.decode(sum(d.bounds) / 2)
                                                        for d in self.space.dimensions)]))
        self._cache = {}
        self._seeds = self._trial_seeds(self.seed, self.replications)
        # Tagged SeedSequences isolate clean-up randomness from DE's stream.
        self.selection_seed = int(trace_rng(seed, 'cimba.optimize.selection').integers(0, 1 << 63))
        self.estimation_seed = int(trace_rng(seed, 'cimba.optimize.estimation').integers(0, 1 << 63))

    @staticmethod
    def _trial_seeds(seed, replications):
        return np.asarray([trial_seed(seed, r) for r in range(replications)], dtype=np.uint64)

    def describe(self):
        rows = []
        for instance in self.assembly.instances:
            for field in instance.schema.fields:
                value = instance.values[field.name]
                rows.append({'instance': instance.label, 'field': field.name,
                             'kind': field.kind,
                             'domain': value.describe() if isinstance(value, Decision) else None,
                             'source': value.describe() if field.kind in {'input', 'series'} else None,
                             'objective': 'minimize' if self.direction == 1 else 'maximize'})
        return rows

    def experiment(self, values):
        """Return a plain Experiment at these values without editing the model."""
        candidate = self.space.candidate(values)
        design = self.space.design([candidate])
        # Bind the entire saved configuration while Experiment takes its own
        # snapshot, including fields edited or applied since study creation.
        saved = [(cast(Model, instance.model), dict(instance.model.__dict__))
                 for instance in self.assembly.instances]
        try:
            for instance in self.assembly.instances:
                for field in instance.schema.fields:
                    object.__setattr__(instance.model, field.name,
                                       design.points[0].bindings[instance.model, field.name])
            return Experiment(self.model, replications=self.replications,
                              window=self.window, seed=self.seed)
        finally:
            for model, state in saved:
                namespace = cast(dict[str, Any], model.__dict__)
                namespace.clear()
                namespace.update(state)

    @staticmethod
    def _options(workers, input_memory, on_failure):
        if workers is not None:
            _positive_integer(workers, 'workers')
        _positive_integer(input_memory, 'input_memory')
        if on_failure not in {'reject', 'ignore', 'raise'}:
            raise ExperimentConfigError("on_failure must be reject, ignore or raise")

    def _batch(self, candidates, seeds, *, phase, generation, proposed, cached,
               workers, input_memory, on_failure):
        started = monotonic()
        design = self.space.design(candidates)
        trial_seeds = np.broadcast_to(seeds, (len(candidates), len(seeds))).copy()
        results = self.snapshot.run(design, trial_seeds, workers=workers,
                                    input_memory=input_memory, captures=phase != 'search')
        context = f"{phase}, generation {generation}"
        try:
            raw = np.asarray(self.objective(results))
            if raw.ndim == 0 or raw.shape == (len(candidates),):
                raise ValueError("return per-trial values shaped (points, replications); a scalar or per-point mean loses pairing")
            if raw.dtype.kind not in 'biuf':
                raise ValueError("objective values must be real numbers")
            values = np.broadcast_to(raw, results.failed.shape).astype(np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise OptimizationError(f"objective in {context}: {exc}") from exc
        failed = results.failed.copy()
        reasons = results.failure_reasons.astype(object)
        invalid = ~failed & ~np.isfinite(values)
        failed |= invalid
        reasons[invalid] = 'objective is not finite'
        values[failed] = np.nan
        estimates = summary(values)
        evaluations = tuple(Evaluation(
            self.space.values(candidate), values[i].copy(), failed[i].copy(),
            tuple(str(x) for x in reasons[i, failed[i]]), Estimate.from_summary(estimates, i))
            for i, candidate in enumerate(candidates))
        if on_failure == 'raise':
            self._raise_failures(evaluations, context)
        wall = monotonic() - started
        # A small input_memory or an input extension may split logical work
        # into several native calls. Report each call, counting new candidates
        # once; host preparation and objective work belong to the first row.
        rows = []
        for index, batch in enumerate(results.meta.native_batches):
            native_wall = batch['native_wall_time']
            rows.append(BatchRow(
                phase, generation, proposed if index == 0 else 0,
                cached if index == 0 else 0, len(candidates) if index == 0 else 0,
                batch['candidates'], batch['trials'], native_wall,
                wall - results.meta.native_wall_time if index == 0 else 0.,
                batch['native_cpu_time'] / native_wall if native_wall else 0., 1,
                results.meta.compile.misses if index == 0 else 0,
                results.meta.compile.wall_time if index == 0 else 0.))
        return evaluations, results, tuple(rows)

    @staticmethod
    def _raise_failures(evaluations, context):
        for evaluation in evaluations:
            if evaluation.failed.any():
                raise TrialsFailed(f"{context}, candidate {dict(evaluation.values)}: {evaluation.reasons[0]}")

    def _evaluate(self, candidates, *, generation, workers, input_memory, on_failure):
        unique = tuple(dict.fromkeys(candidates))
        new = tuple(candidate for candidate in unique if candidate not in self._cache)
        batch = None
        if new:
            evaluations, _, batch = self._batch(
                new, self._seeds, phase='search', generation=generation,
                proposed=len(candidates), cached=len(candidates) - len(new),
                workers=workers, input_memory=input_memory, on_failure=on_failure)
            self._cache.update(zip(new, evaluations))
        evaluations = tuple(self._cache[candidate] for candidate in candidates)
        if on_failure == 'raise':
            self._raise_failures(evaluations, f"search, generation {generation}")
        seen = set(self._cache) - set(new)
        history = []
        for candidate, evaluation in zip(candidates, evaluations):
            estimate = evaluation.estimate
            history.append(HistoryRow(generation, evaluation.values, estimate.mean,
                                      estimate.std, estimate.n, int(evaluation.failed.sum()),
                                      evaluation.reasons, candidate in seen))
            seen.add(candidate)
        return evaluations, history, batch

    def evaluate(self, candidates, *, workers=None, input_memory=1 << 30, on_failure='reject'):
        """Evaluate explicit decision mappings in one batch, reusing search samples."""
        self._options(workers, input_memory, on_failure)
        candidates = tuple(self.space.candidate(values) for values in candidates)
        if not candidates:
            raise ValueError("evaluate needs at least one candidate")
        return self._evaluate(candidates, generation=-1, workers=workers,
                              input_memory=input_memory, on_failure=on_failure)[0]

    def run(self, *, evaluations=None, optimizer=None, initial=None, finalists=5,
            validation=None, workers=None, input_memory=1 << 30,
            on_failure='reject', progress=None):
        self._options(workers, input_memory, on_failure)
        finalists = _positive_integer(finalists, 'finalists')
        validation = max(100, 4 * self.replications) if validation is None else _positive_integer(validation, 'validation', 2)
        if evaluations is not None:
            evaluations = _positive_integer(evaluations, 'evaluations')
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable")
        optimizer = optimizer or DifferentialEvolution()
        settings = optimizer.resolved(len(self.space.dimensions), self.replications, workers or cpu_count() or 1)
        population_size = optimizer.population_size(len(self.space.dimensions), settings['popsize'])
        if evaluations is not None and evaluations < population_size:
            raise ExperimentConfigError(f"evaluations must cover the initial population ({population_size})")
        started = monotonic()
        history, batches = [], []
        proposed_candidates = {}
        generation = -1
        simulated = 0
        stopped = None

        def remaining():
            return float('inf') if evaluations is None else evaluations - simulated

        def evaluate(vectors):
            nonlocal generation, simulated, stopped
            if generation >= 0 and remaining() < population_size:
                stopped = 'budget'
                raise SearchStopped()
            generation += 1
            candidates = self.space.decode(vectors)
            records, rows, batch = self._evaluate(candidates, generation=generation,
                                                 workers=workers, input_memory=input_memory,
                                                 on_failure=on_failure)
            history.extend(rows)
            proposed_candidates.update(zip(candidates, records))
            if batch is not None:
                simulated += sum(row.simulated for row in batch)
                batches.extend(batch)
            energies = np.asarray([record.energy(self.direction, on_failure) for record in records])
            if generation == 0 and not np.isfinite(energies).any():
                reason = next((reason for record in records for reason in record.reasons), 'no finite objective')
                raise OptimizationError(f"every candidate in the initial population was rejected: {reason}")
            return energies

        def callback(intermediate_result):
            nonlocal stopped
            eligible = [r for r in proposed_candidates.values() if isfinite(r.energy(self.direction, on_failure))]
            if progress is not None and progress(Progress(generation, simulated, min(
                    eligible, key=lambda r: r.energy(self.direction, on_failure)))):
                stopped = 'progress'
                return True
            if remaining() < population_size:
                stopped = 'budget'
                return True
            return False

        reason = optimizer.search(self.space, evaluate, trace_rng(self.seed, 'cimba.optimize.de'),
                                  settings=settings, initial=initial, callback=callback)
        ranked = sorted((candidate for candidate, record in proposed_candidates.items()
                         if isfinite(record.energy(self.direction, on_failure))),
                        key=lambda candidate: proposed_candidates[candidate].energy(self.direction, on_failure))[:finalists]
        selection_seeds = self._trial_seeds(self.selection_seed, validation)
        selection, _, selection_batch = self._batch(
            ranked, selection_seeds, phase='selection', generation=None,
            proposed=len(ranked), cached=0, workers=workers, input_memory=input_memory,
            on_failure=on_failure)
        energies = [record.energy(self.direction, on_failure) for record in selection]
        if not any(isfinite(energy) for energy in energies):
            first = next((reason for record in selection for reason in record.reasons), 'no finite objective')
            raise OptimizationError(f"every finalist was rejected on selection seeds: {first}")
        chosen = int(np.argmin(energies))
        estimation, results, estimation_batch = self._batch(
            [ranked[chosen]], self._trial_seeds(self.estimation_seed, validation),
            phase='estimation', generation=None, proposed=1, cached=0,
            workers=workers, input_memory=input_memory, on_failure=on_failure)
        # Under reject, an independently failed estimate cannot be published as
        # a successful answer; ignore remains an explicit survivorship choice.
        if not isfinite(estimation[0].energy(self.direction, on_failure)):
            first = estimation[0].reasons[0] if estimation[0].reasons else 'no finite objective'
            raise OptimizationError(f"chosen candidate failed on estimation seeds: {first}")
        samples = np.asarray([record.samples for record in selection])
        confidence = 1 - 0.05 / max(1, len(ranked) - 1)
        reports = tuple(Finalist(
            self.space.values(candidate), proposed_candidates[candidate].estimate.mean,
            selection[i].estimate, compare(samples, a=chosen, b=i, confidence=confidence),
            not isfinite(energies[i])) for i, candidate in enumerate(ranked))
        batches.extend(selection_batch)
        batches.extend(estimation_batch)
        meta = RunMeta({
            'trials': {'search': simulated * self.replications,
                       'selection': len(ranked) * validation, 'estimation': validation},
            'evaluations': simulated, 'generations': max(0, generation),
            'wall_time': monotonic() - started, 'stop_reason': stopped or reason,
            'optimizer': settings, 'population_size': population_size,
            'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
            'workers': workers, 'seed': self.seed, 'selection_seed': self.selection_seed,
            'estimation_seed': self.estimation_seed, 'replications': self.replications,
            'validation': validation, 'on_failure': on_failure,
            'compile': {'misses': sum(batch.compile_misses for batch in batches),
                        'wall_time': sum(batch.compile_wall_time for batch in batches)},
        })
        return Optimum(estimation[0].values, estimation[0].estimate, reports, results,
                       Table(tuple(history)), Table(tuple(batches)), meta, self.space.bindings)
