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
    noisy = SERIES + np.random.default_rng(0).normal(0.0, 0.5, SERIES.size)
    lumpy = np.where(np.random.default_rng(1).random(60) < 0.3,
                     np.random.default_rng(2).integers(1, 5, 60), 0.0)
    for gen in (bootstrap.iid(SERIES, 50),
                bootstrap.moving_block(SERIES, 50, block=5),
                bootstrap.circular_block(SERIES, 50, block=5),
                bootstrap.stationary(SERIES, 50, mean_block=5),
                bootstrap.residual(noisy, 50),
                bootstrap.residual(noisy, 50, mean_block=4),
                bootstrap.wild(noisy, length=50),
                bootstrap.sieve(noisy, 50, order=2),
                bootstrap.intermittent(lumpy, 50),
                bootstrap.joint({"a": noisy}, 50, name="d",
                                mean_block=5)["a"]):
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


def test_residual_extrapolates_a_pure_trend():
    data = 3.0 + 0.5 * np.arange(40.0)  # exactly linear, zero residuals
    gen = bootstrap.residual(data, length=60, trend=1)
    out = gen(rng(1))
    assert np.allclose(out, 3.0 + 0.5 * np.arange(60.0))


def test_residual_preserves_a_seasonal_shape():
    period = 6
    cycles = 20
    seasonal = np.tile([0.0, 4.0, 8.0, 4.0, 0.0, -4.0], cycles)
    noise = np.random.default_rng(0).normal(0.0, 0.1, seasonal.size)
    gen = bootstrap.residual(seasonal + noise, length=period * 4,
                             trend=0, period=period)
    out = gen(rng(2))
    # Phase means of the output track the true seasonal pattern
    phases = out.reshape(4, period).mean(axis=0)
    assert np.allclose(phases - phases.mean(),
                       np.array([0.0, 4.0, 8.0, 4.0, 0.0, -4.0]) - 2.0,
                       atol=1.0)


def test_period_auto_detects_a_clean_cycle():
    period = 8
    t = np.arange(period * 12, dtype=np.float64)
    data = 10.0 + 5.0 * np.sin(2 * np.pi * t / period)
    data += np.random.default_rng(0).normal(0.0, 0.2, t.size)
    gen = bootstrap.residual(data, length=period * 4, trend=0,
                             period="auto")
    out = gen(rng(3))
    # If the period was found, the output carries the sinusoid shape
    expected = 5.0 * np.sin(2 * np.pi * np.arange(period) / period)
    phases = out.reshape(4, period).mean(axis=0)
    assert np.allclose(phases - phases.mean(), expected, atol=1.5)


def test_period_auto_falls_back_on_white_noise():
    data = np.random.default_rng(0).normal(0.0, 1.0, 128)
    gen = bootstrap.residual(data, length=64, trend=0, period="auto")
    out = gen(rng(4))
    # No seasonal structure claimed: values are mean + resampled noise
    assert np.isin(np.round(out - data.mean(), 10),
                   np.round(data - data.mean(), 10)).all()


def test_trend_auto_picks_up_a_strong_trend():
    t = np.arange(100.0)
    trending = 2.0 * t + np.random.default_rng(0).normal(0.0, 0.5, 100)
    flat = np.random.default_rng(0).normal(5.0, 0.5, 100)
    out_trend = bootstrap.residual(trending, 150, trend="auto")(rng(5))
    out_flat = bootstrap.residual(flat, 150, trend="auto")(rng(5))
    # Extrapolation continues the trend in one case and stays flat in the other
    assert out_trend[140:].mean() > out_trend[:10].mean() + 200.0
    assert abs(out_flat[140:].mean() - 5.0) < 1.0


def test_wild_rademacher_only_flips_residual_signs():
    data = np.random.default_rng(0).normal(0.0, 2.0, 50)
    gen = bootstrap.wild(data, trend=0)
    out = gen(rng(6))
    assert out.shape == (50,)
    resid = data - data.mean()
    assert np.allclose(np.abs(out - data.mean()), np.abs(resid))


def test_wild_default_length_is_the_series_length():
    data = np.random.default_rng(0).normal(0.0, 1.0, 33)
    assert bootstrap.wild(data)(rng(7)).shape == (33,)


def test_sieve_reproduces_ar1_dependence():
    phi = 0.7
    g = np.random.default_rng(1)
    x = np.zeros(400)
    for i in range(1, 400):
        x[i] = phi * x[i - 1] + g.normal()
    gen = bootstrap.sieve(x, length=4000)
    out = gen(rng(8))
    assert out.shape == (4000,)
    centered = out - out.mean()
    lag1 = np.dot(centered[:-1], centered[1:]) / np.dot(centered, centered)
    assert abs(lag1 - phi) < 0.15


def test_sieve_explicit_order_and_validation():
    data = np.random.default_rng(0).normal(0.0, 1.0, 64)
    out = bootstrap.sieve(data, 100, order=3)(rng(9))
    assert out.shape == (100,)
    with pytest.raises(ValueError, match="order must be"):
        bootstrap.sieve(data, 100, order=64)
    with pytest.raises(ValueError, match="at least 8"):
        bootstrap.sieve(np.zeros(5), 10)


def test_residual_wild_validation_errors():
    data = np.random.default_rng(0).normal(0.0, 1.0, 40)
    with pytest.raises(ValueError, match="period must be"):
        bootstrap.residual(data, 10, period=30)
    with pytest.raises(ValueError, match="trend must be"):
        bootstrap.residual(data, 10, trend="cubic")
    with pytest.raises(ValueError, match="mean_block must be"):
        bootstrap.residual(data, 10, mean_block=0.5)
    with pytest.raises(ValueError, match="weights must be"):
        bootstrap.wild(data, weights="bogus")


def test_nonnegative_clips_model_based_outputs():
    g = np.random.default_rng(0)
    data = np.maximum(5.0 - 0.08 * np.arange(60.0) + g.normal(0, 0.5, 60),
                      0.0)
    # Extrapolating the downward trend crosses zero without the clip
    kwargs = dict(trend=1, start=60)
    assert (bootstrap.residual(data, 100, **kwargs)(rng(1)) < 0).any()
    for gen in (bootstrap.residual(data, 100, nonnegative=True, **kwargs),
                bootstrap.wild(data, length=100, nonnegative=True, **kwargs),
                bootstrap.sieve(data, 100, order=1, nonnegative=True,
                                **kwargs)):
        assert (gen(rng(1)) >= 0.0).all()


def test_start_continues_the_trend_beyond_the_data():
    data = 3.0 + 0.5 * np.arange(40.0)  # exactly linear, zero residuals
    out = bootstrap.residual(data, 20, trend=1, start=40)(rng(1))
    assert np.allclose(out, 3.0 + 0.5 * np.arange(40.0, 60.0))
    with pytest.raises(ValueError, match="start must be"):
        bootstrap.residual(data, 20, start=-1)


def test_start_keeps_the_seasonal_phase():
    period, cycles = 6, 20
    pattern = np.array([0.0, 4.0, 8.0, 4.0, 0.0, -4.0])
    data = np.tile(pattern, cycles)
    data += np.random.default_rng(0).normal(0.0, 0.05, data.size)
    gen = bootstrap.residual(data, period * 4, trend=0, period=period,
                             start=data.size)
    out = gen(rng(2))
    phases = out.reshape(4, period).mean(axis=0)
    # start = n is a multiple of the period, so phase 0 aligns
    assert np.allclose(phases - phases.mean(), pattern - pattern.mean(),
                       atol=1.0)


def _mean_run_length(occurs):
    runs, current = [], 0
    for hit in occurs:
        if hit:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return np.mean(runs) if runs else 0.0


def test_intermittent_preserves_zero_fraction_and_sizes():
    g = np.random.default_rng(3)
    data = np.where(g.random(300) < 0.3,
                    g.integers(1, 6, size=300).astype(float), 0.0)
    out = bootstrap.intermittent(data, 3000)(rng(4))
    assert out.shape == (3000,)
    assert abs((out == 0).mean() - (data == 0).mean()) < 0.08
    nonzero = out[out != 0]
    assert np.isin(nonzero, data[data != 0]).all()


def test_intermittent_preserves_demand_clustering():
    # Bursty source: two-state chain with sticky demand periods
    g = np.random.default_rng(5)
    occurs = np.zeros(600, dtype=bool)
    s = 0
    for i in range(600):
        s = 1 if g.random() < (0.85 if s else 0.08) else 0
        occurs[i] = s
    data = np.where(occurs, g.integers(1, 4, 600).astype(float), 0.0)
    out = bootstrap.intermittent(data, 6000)(rng(6))
    # Sticky chain: mean demand run length well above the ~1.5 an
    # i.i.d. occurrence process with the same density would give
    assert _mean_run_length(out != 0) > 2.5


def test_intermittent_jitter_leaves_the_observed_size_set():
    g = np.random.default_rng(7)
    data = np.where(g.random(200) < 0.4,
                    g.integers(2, 6, size=200).astype(float), 0.0)
    out = bootstrap.intermittent(data, 2000, jitter=True)(rng(8))
    nonzero = out[out != 0]
    assert (nonzero > 0).all()
    assert not np.isin(nonzero, data[data != 0]).all()


def test_intermittent_validation():
    with pytest.raises(ValueError, match="nonzero and 1 zero"):
        bootstrap.intermittent(np.zeros(20), 10)  # all zero
    with pytest.raises(ValueError, match="nonzero and 1 zero"):
        bootstrap.intermittent(np.ones(20), 10)  # never zero


def test_joint_generators_share_index_draws():
    a = np.random.default_rng(9).normal(size=80)
    b = 2.0 * a + 1.0  # exact affine relation survives shared indices
    gens = bootstrap.joint({"a": a, "b": b}, 200, name="d", mean_block=5)
    assert gens["a"].trace_rng_name == "joint:d"
    out_a = gens["a"](rng(10))  # equal rngs, as experiment() provides
    out_b = gens["b"](rng(10))
    assert np.allclose(out_b, 2.0 * out_a + 1.0)


def test_joint_validation():
    with pytest.raises(ValueError, match="same length"):
        bootstrap.joint({"a": np.zeros(5), "b": np.zeros(6)}, 10,
                        name="x", mean_block=2)
    with pytest.raises(ValueError, match="non-empty mapping"):
        bootstrap.joint({}, 10, name="x", mean_block=2)
    with pytest.raises(TypeError):
        bootstrap.joint({"a": np.zeros(5)}, 10, mean_block=2)  # no name


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
