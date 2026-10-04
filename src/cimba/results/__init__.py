"""Immutable, object-addressed outcomes detached from native trial blocks."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


def _frozen_provenance(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen_provenance(item)
                                 for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_frozen_provenance(item) for item in value)
    return value


@dataclass(frozen=True, eq=False)
class Samples(NDArrayOperatorsMixin):
    values: np.ndarray

    def __post_init__(self):
        self.values.flags.writeable = False

    def __array__(self, dtype=None, copy=None):
        if copy:
            return np.array(self.values, dtype=dtype, copy=True)
        return np.asarray(self.values, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(x.values if isinstance(x, Samples) else x for x in inputs)
        if "out" in kwargs:
            kwargs["out"] = tuple(x.values if isinstance(x, Samples) else x
                                  for x in kwargs["out"])
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __getitem__(self, index):
        return self.values[index]

    @property
    def shape(self):
        return self.values.shape


@dataclass(frozen=True)
class InputRecord:
    source: tuple[MappingProxyType, ...]
    consumed: np.ndarray
    extended: np.ndarray
    rows_by_trial: Any

    def __post_init__(self):
        object.__setattr__(self, "source", _frozen_provenance(self.source))
        self.consumed.flags.writeable = False
        self.extended.flags.writeable = False

    def rows(self, point: int, replication: int) -> np.ndarray:
        values = np.asarray(self.rows_by_trial(point, replication)).copy()
        values.flags.writeable = False
        return values


@dataclass(frozen=True)
class Signal:
    """Captured series addressed by design point and replication."""

    trials: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...]

    def __post_init__(self):
        for point in self.trials:
            for time, value in point:
                time.flags.writeable = False
                value.flags.writeable = False

    def trial(self, point: int, replication: int):
        """Return read-only ``(times, values)`` arrays for one trial."""
        return self.trials[point][replication]

    @classmethod
    def join(cls, chunks: tuple["Signal", ...]) -> "Signal":
        return cls(tuple(
            tuple(sample for chunk in chunks for sample in chunk.trials[point])
            for point in range(len(chunks[0].trials))))


class InstanceResults:
    def __init__(self, fields: dict[str, Any]):
        self._fields = MappingProxyType(fields)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class RunMeta(Mapping):
    def __init__(self, values: Mapping):
        self._values = MappingProxyType({
            name: RunMeta(value) if isinstance(value, Mapping) else _frozen_provenance(value)
            for name, value in values.items()})

    def __getitem__(self, name) -> Any:
        return self._values[name]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getattr__(self, name) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class Results:
    by_object: MappingProxyType
    failed: np.ndarray
    failure_reasons: np.ndarray
    seeds: np.ndarray
    meta: RunMeta
    design: Any

    def __post_init__(self):
        if not isinstance(self.meta, RunMeta):
            object.__setattr__(self, "meta", RunMeta(self.meta))
        for array in (self.failed, self.failure_reasons, self.seeds):
            array.flags.writeable = False

    def __getitem__(self, model):
        return self.by_object[model]

    def levels(self, sweep):
        return self.design.levels(sweep)

    def to_table(self):
        rows = []
        points, replications = self.failed.shape
        for point in range(points):
            for replication in range(replications):
                row = {"point": point, "replication": replication,
                       "seed": int(self.seeds[point, replication]),
                       "failed": bool(self.failed[point, replication]),
                       "reason": str(self.failure_reasons[point, replication])}
                for model, instance in self.by_object.items():
                    for name, value in instance._fields.items():
                        if isinstance(value, Samples):
                            row[f"{type(model).__name__}.{name}"] = value[point, replication]
                rows.append(row)
        return rows
