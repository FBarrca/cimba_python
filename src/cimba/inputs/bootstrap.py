"""Prefix-stable bootstrap sources for recorded sequences and panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .sources import GeneratedRows, InputError, reference, row_source


def _block_size(block: int, size: int) -> int:
    if not isinstance(block, int) or not 1 <= block <= size:
        raise InputError(f"block must be in [1, {size}]")
    return block


def stationary_indices(
    rng: np.random.Generator, size: int, length: int, mean_block: float
) -> NDArray[np.intp]:
    if not np.isfinite(mean_block) or mean_block < 1:
        raise InputError("mean_block must be finite and >= 1")
    out = np.empty(length, dtype=np.intp)
    p = 1.0 / mean_block
    offset = 0
    while offset < length:
        start = int(rng.integers(0, size))
        run = int(rng.geometric(p))
        stop = min(length, offset + run)
        out[offset:stop] = (start + np.arange(stop - offset)) % size
        offset = stop
    return out


def iid(data: ArrayLike) -> GeneratedRows:
    values, _ = reference(data, "iid")

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        return values[rng.integers(0, len(values), size=length)]

    return row_source("bootstrap.iid", values, draw)


def moving_block(data: ArrayLike, *, block: int) -> GeneratedRows:
    values, _ = reference(data, "moving_block")
    block = _block_size(block, len(values))
    offsets = np.arange(block)

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        starts = rng.integers(0, len(values) - block + 1,
                              size=(length + block - 1) // block)
        return values[(starts[:, None] + offsets).ravel()[:length]]

    return row_source("bootstrap.moving_block", values, draw,
                      parameters={"block": block})


def circular_block(data: ArrayLike, *, block: int) -> GeneratedRows:
    values, _ = reference(data, "circular_block")
    block = _block_size(block, len(values))
    offsets = np.arange(block)

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        starts = rng.integers(0, len(values), size=(length + block - 1) // block)
        return values[(starts[:, None] + offsets).ravel()[:length] % len(values)]

    return row_source("bootstrap.circular_block", values, draw,
                      parameters={"block": block})


def stationary(data: ArrayLike, *, mean_block: float) -> GeneratedRows:
    values, _ = reference(data, "stationary")
    if not np.isfinite(mean_block) or mean_block < 1:
        raise InputError("mean_block must be finite and >= 1")

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        return values[stationary_indices(rng, len(values), length, mean_block)]

    return row_source("bootstrap.stationary", values, draw,
                      parameters={"mean_block": mean_block})


def intermittent(data: ArrayLike, *, jitter: bool = False) -> GeneratedRows:
    values, _ = reference(data, "intermittent")
    occurs = values != 0
    sizes = values[occurs]
    if len(sizes) < 2 or len(sizes) == len(values):
        raise InputError("intermittent needs at least two nonzero and one zero value")
    state = occurs.astype(np.int64)
    from_zero = state[:-1] == 0
    n0 = int(from_zero.sum())
    n1 = len(values) - 1 - n0
    p01 = (float(state[1:][from_zero].sum()) + 1.0) / (n0 + 2.0)
    p11 = (float(state[1:][~from_zero].sum()) + 1.0) / (n1 + 2.0)
    p_start = len(sizes) / len(values)

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        # Draw in time order: extending a row cannot shift later RNG calls.
        out = np.zeros(length, dtype=np.float64)
        state_now = rng.random() < p_start
        for i in range(length):
            state_now = rng.random() < (p11 if state_now else p01)
            if state_now:
                value = float(sizes[rng.integers(0, len(sizes))])
                if jitter:
                    moved = value + rng.standard_normal() * np.sqrt(abs(value))
                    if moved > 0:
                        value = moved
                out[i] = value
        return out

    return row_source("bootstrap.intermittent", values, draw,
                      parameters={"jitter": jitter})


class JointMembers:
    """Members of a panel resample, indexed like the supplied panel."""

    def __init__(self, panel: Mapping[Any, ArrayLike] | Sequence[ArrayLike],
                 mean_block: float, tag: str | None):
        if isinstance(panel, Mapping):
            items = list(panel.items())
            self._mapping = True
        else:
            items = list(enumerate(panel))
            self._mapping = False
        if not items:
            raise InputError("joint panel must be nonempty")
        arrays = [reference(value, f"joint[{key}]")[0] for key, value in items]
        if len({len(x) for x in arrays}) != 1:
            raise InputError("joint panel columns must have equal length")
        fingerprint = sha256(b"".join(x.tobytes() for x in arrays)).hexdigest()
        self.tag = tag or f"joint:{fingerprint}"
        self._members: dict[Any, GeneratedRows] = {}
        for (key, _), values in zip(items, arrays):
            def draw(rng: np.random.Generator, length: int,
                     column: NDArray[np.float64] = values) -> NDArray[np.float64]:
                return column[stationary_indices(rng, len(column), length,
                                                 mean_block)]

            self._members[key] = row_source(
                "bootstrap.joint", values, draw,
                parameters={"mean_block": mean_block, "member": str(key),
                            "panel_sha256": fingerprint}, tag=self.tag,
            )

    def __getitem__(self, key: Any) -> GeneratedRows:
        return self._members[key]

    def __iter__(self):
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)


def joint(panel: Mapping[Any, ArrayLike] | Sequence[ArrayLike], *,
          mean_block: float, tag: str | None = None) -> JointMembers:
    return JointMembers(panel, mean_block, tag)
