"""Annotation-driven decision domains and explicit batch design points."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from types import MappingProxyType

import numpy as np

from cimba.experiments import Design, DesignPoint
from cimba.inputs.sources import DistributionSource
from cimba.modeling import Decision, Sweep
from cimba.schema import Assembly, ModelDefinitionError, scalar_value


@dataclass(frozen=True)
class Dimension:
    decision: Decision
    value_type: type
    path: str

    def __post_init__(self):
        d = self.decision
        if self.value_type not in (float, int, bool):
            raise ModelDefinitionError(f"{self.path}: decisions require Param[float, int or bool]")
        if self.value_type is int:
            if d.log:
                raise ModelDefinitionError(f"{self.path}: log integer decisions are not supported")
            if any(int(x) != x for x in (d.low, d.high, d.step or 1)):
                raise ModelDefinitionError(f"{self.path}: integer decision bounds and step must be integral")
            if not -(1 << 53) < d.low < d.high < (1 << 53):
                raise ModelDefinitionError(f"{self.path}: integer decision bounds must be exactly representable as float64")
        if self.value_type is bool and (d.low != 0 or d.high != 1 or d.step is not None or d.log):
            raise ModelDefinitionError(f"{self.path}: boolean decisions require False, True without step or log")

    @property
    def bounds(self):
        d = self.decision
        if d.step is not None:
            return 0.0, float(round((d.high - d.low) / d.step))
        if d.log:
            return log(d.low), log(d.high)
        return float(d.low), float(d.high)

    @property
    def integral(self):
        return self.value_type in (int, bool) or self.decision.step is not None

    def decode(self, coordinate):
        low, high = self.bounds
        value = float(np.clip(coordinate, low, high))
        if self.integral:
            value = float(np.rint(value))
        d = self.decision
        if d.step is not None:
            value = d.low + value * d.step
        elif d.log:
            value = exp(value)
        return self.value_type(np.clip(value, d.low, d.high))

    def encode(self, value):
        d = self.decision
        try:
            numeric = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{self.path}: decision value must be numeric") from exc
        if not np.isfinite(numeric) or not d.low <= numeric <= d.high:
            raise ValueError(f"{self.path}: decision value is outside [{d.low}, {d.high}]")
        if self.value_type in (int, bool) and numeric != round(numeric):
            raise ValueError(f"{self.path}: decision value must be integral")
        if d.step is not None:
            coordinate = (numeric - d.low) / d.step
            if not np.isclose(coordinate, round(coordinate), rtol=0, atol=1e-9):
                raise ValueError(f"{self.path}: decision value must lie on its step lattice")
            return float(round(coordinate))
        return log(numeric) if d.log else numeric


class Space:
    def __init__(self, assembly: Assembly):
        self.assembly = assembly
        self.bindings = assembly.decisions()
        groups = {}
        for instance in assembly.instances:
            for field in instance.schema.fields:
                value = instance.values[field.name]
                path = f"{instance.label}.{field.name}"
                if isinstance(value, Sweep) or (isinstance(value, DistributionSource) and
                        any(isinstance(x, Sweep) for _, x in value.parameters)):
                    raise ModelDefinitionError(f"{path}: sweeps are not supported in Optimization")
                if field.kind in {"param", "state", "constant"} and not isinstance(value, Decision):
                    scalar_value(field, value, path)
        for instance, field, decision in self.bindings:
            groups.setdefault(decision.root, []).append((instance, field, decision))
        if not groups:
            raise ModelDefinitionError("nothing to optimize: assign cb.decision to a Param")
        dimensions = []
        for root, bindings in groups.items():
            direct = [(instance, field) for instance, field, value in bindings if value is root]
            instance, field = direct[0] if direct else bindings[0][:2]
            for other, target in direct[1:]:
                if target.value_type is not field.value_type:
                    raise ModelDefinitionError(
                        f"{other.label}.{target.name}: one decision cannot link fields of different types")
            dimensions.append(Dimension(root, field.value_type, f"{instance.label}.{field.name}"))
        self.dimensions = tuple(dimensions)
        self.decisions = tuple(d.decision for d in self.dimensions)
        self.bounds = tuple(d.bounds for d in self.dimensions)
        self.integrality = tuple(d.integral for d in self.dimensions)

    def decode(self, vectors):
        return tuple(tuple(d.decode(x) for d, x in zip(self.dimensions, vector))
                     for vector in vectors)

    def encode(self, values):
        if set(values) != set(self.decisions):
            raise ValueError("candidate must specify every decision exactly once")
        return np.asarray([d.encode(values[d.decision]) for d in self.dimensions])

    def candidate(self, values):
        """Validate model values without a lossy log/exp round trip."""
        coordinates = self.encode(values)
        return tuple(d.decode(coordinate) if d.decision.step is not None
                     else d.value_type(values[d.decision])
                     for d, coordinate in zip(self.dimensions, coordinates))

    def values(self, candidate):
        base = dict(zip(self.decisions, candidate))
        for _, _, decision in self.bindings:
            if decision not in base:
                base[decision] = decision.value(base[decision.root])
        return MappingProxyType(base)

    def design(self, candidates):
        points = []
        for index, candidate in enumerate(candidates):
            levels = dict(zip(self.decisions, candidate))
            bindings = {}
            for instance in self.assembly.instances:
                for field in instance.schema.fields:
                    value = instance.values[field.name]
                    if isinstance(value, Decision):
                        value = value.value(levels[value.root])
                    if field.kind in {"param", "state", "constant"}:
                        value = scalar_value(field, value, f"{instance.label}.{field.name}")
                    bindings[instance.model, field.name] = value
            points.append(DesignPoint(index, MappingProxyType(levels), MappingProxyType(bindings)))
        return Design(MappingProxyType(dict.fromkeys(self.decisions)), tuple(points))
