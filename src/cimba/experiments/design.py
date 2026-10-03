"""Expand object-bound sweeps and derive trial and input seeds."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from types import MappingProxyType
from typing import Any

import numpy as np

from cimba.inputs.sources import DistributionSource
from cimba.modeling import Model, Sweep
from cimba.schema import Assembly


@dataclass(frozen=True)
class DesignPoint:
    index: int
    levels: MappingProxyType
    bindings: MappingProxyType[tuple[Model, str], Any]


@dataclass(frozen=True)
class Design:
    axes: MappingProxyType
    points: tuple[DesignPoint, ...]

    @classmethod
    def of(cls, assembly: Assembly) -> "Design":
        axes: dict[int, Sweep] = {}
        for instance in assembly.instances:
            for field in instance.schema.fields:
                value = instance.values[field.name]
                if isinstance(value, Sweep):
                    previous = axes.setdefault(value.axis, value)
                    if len(previous.values) != len(value.values):
                        raise ValueError("linked sweeps must have equal lengths")
                elif isinstance(value, DistributionSource):
                    for _, parameter in value.parameters:
                        if isinstance(parameter, Sweep):
                            previous = axes.setdefault(parameter.axis, parameter)
                            if len(previous.values) != len(parameter.values):
                                raise ValueError("linked sweeps must have equal lengths")
        axis_ids = tuple(axes)
        points: list[DesignPoint] = []
        for positions in product(*(range(len(axes[axis].values)) for axis in axis_ids)):
            levels = dict(zip(axis_ids, positions))
            bindings: dict[tuple[Model, str], Any] = {}
            for instance in assembly.instances:
                for field in instance.schema.fields:
                    value = instance.values[field.name]
                    if isinstance(value, Sweep):
                        value = value.values[levels[value.axis]]
                    if isinstance(value, DistributionSource):
                        parameters = tuple(
                            (key, parameter.values[levels[parameter.axis]]
                             if isinstance(parameter, Sweep) else parameter)
                            for key, parameter in value.parameters
                        )
                        value = replace(value, parameters=parameters)
                    bindings[(instance.model, field.name)] = value
            points.append(DesignPoint(len(points), MappingProxyType(levels),
                                      MappingProxyType(bindings)))
        return cls(MappingProxyType(axes), tuple(points))

    def levels(self, sweep: Sweep) -> tuple[Any, ...]:
        if sweep.axis not in self.axes:
            raise KeyError("sweep does not belong to this design")
        return tuple(sweep.values[point.levels[sweep.axis]] for point in self.points)


def trial_seed(seed: int, replication: int, point: int = 0,
               *, seeding: str = "common") -> int:
    if seeding not in {"common", "independent"}:
        raise ValueError("seeding must be 'common' or 'independent'")
    entropy = [int(seed), int(replication)]
    if seeding == "independent":
        entropy.append(int(point))
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])
