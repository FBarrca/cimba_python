import math

import pytest

import cimba


def test_random_generator_is_repeatable_and_distributions_are_in_range():
    assert cimba.seed(0xC0FFEE) == 0xC0FFEE
    assert cimba.current_seed() == 0xC0FFEE
    first = [cimba.random_u64() for _ in range(5)]
    cimba.seed(0xC0FFEE)
    second = [cimba.random_u64() for _ in range(5)]
    assert first == second
    assert cimba.fmix64(123, 456) == cimba.fmix64(123, 456)

    assert 1.0 <= cimba.uniform(1.0, 2.0) <= 2.0
    assert 0.0 <= cimba.triangular(0.0, 0.5, 1.0) <= 1.0
    assert 0.0 <= cimba.pert(0.0, 0.5, 1.0) <= 1.0
    assert 0.0 <= cimba.pert_mod(0.0, 0.5, 1.0, 6.0) <= 1.0
    assert 0.0 <= cimba.beta(2.0, 3.0) <= 1.0
    assert cimba.exponential(1.0) >= 0.0
    assert cimba.gamma(2.0) >= 0.0
    assert 1 <= cimba.dice(1, 6) <= 6
    assert isinstance(cimba.flip(), bool)
    assert isinstance(cimba.bernoulli(0.25), bool)
    assert math.isfinite(cimba.normal())


def test_additional_random_distributions_are_repeatable_and_in_range():
    cimba.seed(0xA11CE)
    first = (
        cimba.lognormal(0.0, 1.0),
        cimba.logistic(0.0, 1.0),
        cimba.cauchy(0.0, 1.0),
        cimba.erlang(2, 1.5),
        cimba.hypoexponential([1.0, 2.0, 3.0]),
        cimba.hyperexponential([1.0, 3.0], [0.25, 0.75]),
        cimba.weibull(2.0, 3.0),
        cimba.pareto(2.0, 1.5),
        cimba.chi_squared(4.0),
        cimba.f_dist(3.0, 7.0),
        cimba.student_t(5.0),
        cimba.rayleigh(2.0),
        cimba.geometric(0.5),
        cimba.binomial(8, 0.5),
        cimba.negative_binomial(3, 0.5),
        cimba.pascal(3, 0.5),
        cimba.poisson(2.5),
        cimba.loaded_dice([0.2, 0.3, 0.5]),
    )
    cimba.seed(0xA11CE)
    second = (
        cimba.lognormal(0.0, 1.0),
        cimba.logistic(0.0, 1.0),
        cimba.cauchy(0.0, 1.0),
        cimba.erlang(2, 1.5),
        cimba.hypoexponential([1.0, 2.0, 3.0]),
        cimba.hyperexponential([1.0, 3.0], [0.25, 0.75]),
        cimba.weibull(2.0, 3.0),
        cimba.pareto(2.0, 1.5),
        cimba.chi_squared(4.0),
        cimba.f_dist(3.0, 7.0),
        cimba.student_t(5.0),
        cimba.rayleigh(2.0),
        cimba.geometric(0.5),
        cimba.binomial(8, 0.5),
        cimba.negative_binomial(3, 0.5),
        cimba.pascal(3, 0.5),
        cimba.poisson(2.5),
        cimba.loaded_dice([0.2, 0.3, 0.5]),
    )

    assert first == second
    for value in first[:12]:
        assert math.isfinite(value)
    assert first[0] >= 0.0
    assert first[3] >= 0.0
    assert first[4] >= 0.0
    assert first[5] >= 0.0
    assert first[6] >= 0.0
    assert first[7] >= 1.5
    assert first[8] >= 0.0
    assert first[9] >= 0.0
    assert first[11] >= 0.0
    assert first[12] >= 1
    assert 0 <= first[13] <= 8
    assert first[14] >= 0
    assert first[15] >= 0
    assert first[16] >= 0
    assert 0 <= first[17] <= 2


def test_loaded_dice_and_alias_sampler_degenerate_probabilities():
    assert cimba.loaded_dice([0.0, 1.0]) == 1

    sampler = cimba.AliasSampler([0.0, 1.0])
    assert len(sampler) == 2
    assert sampler.sample() == 1
    sampler.close()
    sampler.close()
    with pytest.raises(RuntimeError, match="closed"):
        sampler.sample()

    with cimba.AliasSampler([1.0, 0.0]) as scoped:
        assert scoped.sample() == 0
    with pytest.raises(RuntimeError, match="closed"):
        scoped.sample()


def test_discrete_distribution_p_one_boundaries():
    assert cimba.geometric(1.0) == 1
    assert cimba.binomial(3, 1.0) == 3
    assert cimba.negative_binomial(3, 1.0) == 0
    assert cimba.pascal(3, 1.0) == 0


@pytest.mark.parametrize(
    "probabilities",
    [
        [],
        [math.nan],
        [math.inf],
        [-0.1, 1.1],
        [0.2, 0.2],
    ],
)
def test_discrete_probability_validation(probabilities):
    with pytest.raises(ValueError):
        cimba.loaded_dice(probabilities)
    with pytest.raises(ValueError):
        cimba.AliasSampler(probabilities)


def test_sequence_and_unsigned_validation():
    with pytest.raises(ValueError, match="same length"):
        cimba.hyperexponential([1.0, 2.0], [1.0])
    with pytest.raises(OverflowError):
        cimba.erlang(1 << 32, 1.0)
    with pytest.raises(OverflowError):
        cimba.binomial(1 << 32, 0.5)
