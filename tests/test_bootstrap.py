"""Bootstrap trajectory generators: cimba.bootstrap factories."""

import numpy as np
import pytest

from cimba import bootstrap


def rng(seed=0):
    return np.random.default_rng(seed)


SERIES = np.arange(20.0)  # integer values make block structure checkable


def test_iid_resamples_from_the_data():
    gen = bootstrap.iid(SERIES, length=100)
    out = gen(rng())
    assert out.shape == (100,)
    assert out.dtype == np.float64
    assert np.isin(out, SERIES).all()


def test_generators_are_deterministic_per_rng():
    for gen in (bootstrap.iid(SERIES, 50),
                bootstrap.moving_block(SERIES, 50, block=5),
                bootstrap.circular_block(SERIES, 50, block=5),
                bootstrap.stationary(SERIES, 50, mean_block=5)):
        assert np.array_equal(gen(rng(7)), gen(rng(7)))
        assert not np.array_equal(gen(rng(7)), gen(rng(8)))


def test_moving_block_preserves_contiguous_runs():
    block = 5
    gen = bootstrap.moving_block(SERIES, length=23, block=block)
    out = gen(rng(1))
    assert out.shape == (23,)
    # Within a block consecutive values step by exactly 1, and no
    # block wraps (starts never exceed n - block)
    diffs = np.diff(out)
    interior = [d for i, d in enumerate(diffs) if (i + 1) % block != 0]
    assert np.allclose(interior, 1.0)
    assert out.max() <= SERIES.max()


def test_circular_block_wraps_instead_of_truncating():
    block = 8
    gen = bootstrap.circular_block(SERIES, length=400, block=block)
    out = gen(rng(2))
    diffs = np.diff(out)
    interior = [d for i, d in enumerate(diffs) if (i + 1) % block != 0]
    # Steps are +1 within the series or 1 - n at the wrap point
    assert set(np.unique(interior)) <= {1.0, 1.0 - SERIES.size}
    # With 50 blocks of length 8 over 20 points, wraps must occur
    assert (np.array(interior) < 0).any()


def test_stationary_concatenates_wrapped_runs():
    gen = bootstrap.stationary(SERIES, length=300, mean_block=6)
    out = gen(rng(3))
    assert out.shape == (300,)
    assert np.isin(out, SERIES).all()
    # Every step either continues a run (+1, possibly wrapping) or
    # jumps to a fresh uniform block start
    diffs = np.diff(out)
    continues = (diffs == 1.0) | (diffs == 1.0 - SERIES.size)
    # Mean block length ~6 => a substantial majority of steps continue
    assert continues.mean() > 0.5


def test_validation_errors():
    with pytest.raises(ValueError, match="non-empty 1-D"):
        bootstrap.iid([], 10)
    with pytest.raises(ValueError, match="non-empty 1-D"):
        bootstrap.iid(np.zeros((2, 2)), 10)
    with pytest.raises(ValueError, match="length must be"):
        bootstrap.iid(SERIES, 0)
    with pytest.raises(ValueError, match="block must be"):
        bootstrap.moving_block(SERIES, 10, block=0)
    with pytest.raises(ValueError, match="block must be"):
        bootstrap.circular_block(SERIES, 10, block=SERIES.size + 1)
    with pytest.raises(ValueError, match="mean_block must be"):
        bootstrap.stationary(SERIES, 10, mean_block=0.5)
