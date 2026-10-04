"""Immutable optimization estimates, candidate history and batch reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from math import isfinite
from types import MappingProxyType

import numpy as np

from cimba.analysis import Comparison
from cimba.modeling import Decision
from cimba.results import Results, RunMeta


class OptimizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Estimate:
    n: int
    mean: float
    std: float
    lower: float
    upper: float

    @classmethod
    def from_summary(cls, summary, index=0):
        return cls(int(summary.n[index]), *(float(getattr(summary, name)[index])
                    for name in ('mean', 'std', 'lower', 'upper')))


@dataclass(frozen=True)
class Evaluation:
    values: MappingProxyType
    samples: np.ndarray
    failed: np.ndarray
    reasons: tuple[str, ...]
    estimate: Estimate

    def __post_init__(self):
        self.samples.flags.writeable = False
        self.failed.flags.writeable = False

    def energy(self, direction, on_failure):
        if not self.estimate.n or not isfinite(self.estimate.mean) or (on_failure == 'reject' and self.failed.any()):
            return float('inf')
        return direction * self.estimate.mean


@dataclass(frozen=True)
class HistoryRow:
    generation: int
    values: MappingProxyType
    mean: float
    std: float
    n: int
    failed: int
    reasons: tuple[str, ...]
    cached: bool


@dataclass(frozen=True)
class BatchRow:
    phase: str
    generation: int | None
    proposed: int
    cached: int
    simulated: int
    candidates: int
    trials: int
    native_wall_time: float
    host_wall_time: float
    busy_cpus: float
    native_calls: int
    compile_misses: int
    compile_wall_time: float


@dataclass(frozen=True)
class Table(Sequence):
    rows: tuple

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)

    def to_table(self):
        return [{field.name: getattr(row, field.name) for field in fields(row)} for row in self.rows]


@dataclass(frozen=True)
class Finalist:
    values: MappingProxyType
    search_mean: float
    selection: Estimate
    difference: Comparison
    rejected: bool

    @property
    def selection_mean(self):
        return self.selection.mean

    @property
    def indistinguishable(self):
        return not self.rejected and self.difference.lower <= 0 <= self.difference.upper


@dataclass(frozen=True)
class Progress:
    generation: int
    evaluations: int
    best: Evaluation


@dataclass(frozen=True)
class Optimum:
    values: MappingProxyType
    estimate: Estimate
    finalists: tuple[Finalist, ...]
    results: Results
    history: Table
    batches: Table
    meta: RunMeta
    bindings: tuple

    def __getitem__(self, decision: Decision):
        return self.values[decision]

    def apply(self):
        for instance, field, decision in self.bindings:
            setattr(instance.model, field.name, self.values[decision])
