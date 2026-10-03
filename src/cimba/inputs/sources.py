"""Immutable, source-agnostic descriptions of simulation inputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cimba.modeling import Input, Series


class InputError(ValueError):
    """An input cannot be bound or generated as requested."""


def reference(data: ArrayLike, name: str = "source") -> tuple[NDArray[np.float64], str]:
    """Copy and fingerprint a finite, one-dimensional reference series."""
    values = np.array(data, dtype=np.float64, order="C", copy=True)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise InputError(f"{name}: expected a non-empty, finite 1-D series")
    values.flags.writeable = False
    return values, sha256(values.tobytes()).hexdigest()


@runtime_checkable
class Source(Protocol):
    @property
    def on_exhausted(self) -> str: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True, eq=False)
class EmptySource(Input[Any]):
    """The finite, empty binding of an unbound optional Input field."""

    on_exhausted: str = "fail"

    def describe(self) -> dict[str, Any]:
        return {"version": 1, "method": "empty", "length": 0,
                "on_exhausted": self.on_exhausted}


@runtime_checkable
class RowSource(Source, Protocol):
    @property
    def prefix_stable(self) -> bool: ...

    @property
    def max_length(self) -> int | None: ...

    @property
    def length_hint(self) -> int | None: ...

    @property
    def tag(self) -> str | None: ...

    def generate(
        self, rngs: Sequence[np.random.Generator], length: int
    ) -> NDArray[np.float64]: ...


def validate_policy(policy: str, *, extend: bool = False) -> str:
    choices = {"fail", "wrap", "end_trial"}
    if extend:
        choices.add("extend")
    if policy not in choices:
        raise InputError(f"on_exhausted must be one of {sorted(choices)}")
    return policy


@dataclass(frozen=True, eq=False)
class TraceSource(Input[Any], Series[Any]):
    values: NDArray[np.float64]
    fingerprint: str
    on_exhausted: str = "fail"

    def describe(self) -> dict[str, Any]:
        return {
            "version": 1, "method": "trace", "length": len(self.values),
            "sha256": self.fingerprint, "on_exhausted": self.on_exhausted,
        }

    def verify(self) -> None:
        if sha256(self.values.tobytes()).hexdigest() != self.fingerprint:
            raise InputError("trace reference data changed after construction")


def trace(data: ArrayLike, *, on_exhausted: str = "fail") -> TraceSource:
    values, fingerprint = reference(data, "trace")
    return TraceSource(values, fingerprint, validate_policy(on_exhausted))


@dataclass(frozen=True, eq=False)
class DistributionSource(Input[Any], Series[Any]):
    method: str
    parameters: tuple[tuple[str, Any], ...]
    on_exhausted: str = "fail"
    tag: str | None = None

    def __post_init__(self) -> None:
        if self.tag is not None and not isinstance(self.tag, str):
            raise InputError("distribution tag must be a string or None")

    def describe(self) -> dict[str, Any]:
        return {"version": 1, "method": f"dist.{self.method}",
                "parameters": dict(self.parameters), "tag": self.tag}


@dataclass(frozen=True, eq=False)
class GeneratedRows(Input[Any], Series[Any]):
    """A finite row source whose generator consumes RNG draws in prefix order."""

    method: str
    data: NDArray[np.float64]
    fingerprint: str
    parameters: tuple[tuple[str, Any], ...]
    draw: Callable[[np.random.Generator, int], NDArray[np.float64]] = field(repr=False)
    tag: str | None = None
    prefix_stable: bool = True
    max_length: int | None = None
    length_hint: int | None = None
    on_exhausted: str = "extend"

    def __post_init__(self) -> None:
        validate_policy(self.on_exhausted, extend=self.prefix_stable)

    def generate(
        self, rngs: Sequence[np.random.Generator], length: int
    ) -> NDArray[np.float64]:
        if length < 1 or (self.max_length is not None and length > self.max_length):
            raise InputError(f"{self.method}: invalid row length {length}")
        if sha256(self.data.tobytes()).hexdigest() != self.fingerprint:
            raise InputError(f"{self.method}: reference data changed after construction")
        rows = np.empty((len(rngs), length), dtype=np.float64)
        for i, rng in enumerate(rngs):
            row = np.asarray(self.draw(rng, length), dtype=np.float64)
            if row.shape != (length,) or not np.isfinite(row).all():
                raise InputError(f"{self.method}: generator returned invalid row {i}")
            rows[i] = row
        rows.flags.writeable = False
        return rows

    def describe(self) -> dict[str, Any]:
        return {
            "version": 1, "method": self.method,
            "parameters": dict(self.parameters), "sha256": self.fingerprint,
            "tag": self.tag, "prefix_stable": self.prefix_stable,
            "max_length": self.max_length, "on_exhausted": self.on_exhausted,
        }


def row_source(
    method: str, data: ArrayLike,
    draw: Callable[[np.random.Generator, int], NDArray[np.float64]],
    *, parameters: dict[str, Any] | None = None, tag: str | None = None,
    prefix_stable: bool = True, max_length: int | None = None,
    length_hint: int | None = None, on_exhausted: str = "extend",
) -> GeneratedRows:
    values, fingerprint = reference(data, method)
    return GeneratedRows(method, values, fingerprint,
                         tuple(sorted((parameters or {}).items())), draw,
                         tag, prefix_stable, max_length, length_hint,
                         on_exhausted)


def trace_rng(seed: int, tag: str) -> np.random.Generator:
    """Derive a stable stream without consuming another input's RNG."""
    digest = sha256(tag.encode("utf-8")).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32)
    seed_words = [int(seed) & 0xffffffff, (int(seed) >> 32) & 0xffffffff]
    return np.random.default_rng(np.random.SeedSequence(seed_words + words.tolist()))
