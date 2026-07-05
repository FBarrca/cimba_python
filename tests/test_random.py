import math

import pytest
from numba import njit

import cimba
import cimba.sim as sim


def test_flat_random_api_was_removed():
    assert hasattr(cimba, "random")
    assert not hasattr(sim, "random")
    removed = (
        "exponential",
        "uniform",
        "normal",
        "random01",
        "flip",
        "std_normal",
        "std_exponential",
        "std_gamma",
        "std_beta",
        "loaded_dice",
        "pascal",
        "chisquared",
        "std_t",
        "t_dist",
    )
    for name in removed:
        assert not hasattr(cimba, name)
        assert not hasattr(sim, name)
    assert not hasattr(cimba.random, "loaded_dice")
    assert not hasattr(cimba.random, "pascal")


def test_random_generator_is_repeatable_and_distributions_are_in_range():
    assert cimba.random.seed(0xC0FFEE) == 0xC0FFEE
    assert cimba.random.current_seed() == 0xC0FFEE
    first = [cimba.random.random_u64() for _ in range(5)]
    cimba.random.seed(0xC0FFEE)
    second = [cimba.random.random_u64() for _ in range(5)]
    assert first == second
    assert cimba.random.fmix64(123, 456) == cimba.random.fmix64(123, 456)

    assert 1.0 <= cimba.random.uniform(1.0, 2.0) <= 2.0
    assert 0.0 <= cimba.random.triangular(0.0, 0.5, 1.0) <= 1.0
    assert 0.0 <= cimba.random.pert(0.0, 0.5, 1.0) <= 1.0
    assert 0.0 <= cimba.random.pert_mod(0.0, 0.5, 1.0, 6.0) <= 1.0
    assert 0.0 <= cimba.random.beta(2.0, 3.0) <= 1.0
    assert cimba.random.exponential(1.0) >= 0.0
    assert cimba.random.gamma(2.0) >= 0.0
    assert 1 <= cimba.random.dice(1, 6) <= 6
    assert isinstance(cimba.random.bernoulli(0.5), bool)
    assert isinstance(cimba.random.bernoulli(0.25), bool)
    assert math.isfinite(cimba.random.normal())


def test_random_namespace_is_available_in_standalone_njit_helpers():
    @njit
    def draw():
        return (
            cimba.random.uniform(),
            cimba.random.normal(mu=0.0, sigma=1.0),
            cimba.random.exponential(),
            cimba.random.gamma(shape=2.0),
            cimba.random.categorical((0.2, 0.3, 0.5)),
            cimba.random.hyperexponential((1.0, 2.0), (0.25, 0.75)),
            cimba.random.student_t(v=7.0, m=1.0, s=2.0),
            cimba.random.chi_squared(k=4.0),
        )

    values = draw()
    assert 0.0 <= values[0] <= 1.0
    assert values[2] >= 0.0
    assert values[3] >= 0.0
    assert 0 <= values[4] <= 2
    assert values[5] >= 0.0
    for value in (values[1], values[6], values[7]):
        assert math.isfinite(value)


def test_additional_random_distributions_are_repeatable_and_in_range():
    cimba.random.seed(0xA11CE)
    first = (
        cimba.random.lognormal(0.0, 1.0),
        cimba.random.logistic(0.0, 1.0),
        cimba.random.cauchy(0.0, 1.0),
        cimba.random.erlang(2, 1.5),
        cimba.random.hypoexponential([1.0, 2.0, 3.0]),
        cimba.random.hyperexponential([1.0, 3.0], [0.25, 0.75]),
        cimba.random.weibull(2.0, 3.0),
        cimba.random.pareto(2.0, 1.5),
        cimba.random.chi_squared(4.0),
        cimba.random.f_dist(3.0, 7.0),
        cimba.random.student_t(5.0),
        cimba.random.rayleigh(2.0),
        cimba.random.geometric(0.5),
        cimba.random.binomial(8, 0.5),
        cimba.random.negative_binomial(3, 0.5),
        cimba.random.negative_binomial(3, 0.5),
        cimba.random.poisson(2.5),
        cimba.random.categorical([0.2, 0.3, 0.5]),
    )
    cimba.random.seed(0xA11CE)
    second = (
        cimba.random.lognormal(0.0, 1.0),
        cimba.random.logistic(0.0, 1.0),
        cimba.random.cauchy(0.0, 1.0),
        cimba.random.erlang(2, 1.5),
        cimba.random.hypoexponential([1.0, 2.0, 3.0]),
        cimba.random.hyperexponential([1.0, 3.0], [0.25, 0.75]),
        cimba.random.weibull(2.0, 3.0),
        cimba.random.pareto(2.0, 1.5),
        cimba.random.chi_squared(4.0),
        cimba.random.f_dist(3.0, 7.0),
        cimba.random.student_t(5.0),
        cimba.random.rayleigh(2.0),
        cimba.random.geometric(0.5),
        cimba.random.binomial(8, 0.5),
        cimba.random.negative_binomial(3, 0.5),
        cimba.random.negative_binomial(3, 0.5),
        cimba.random.poisson(2.5),
        cimba.random.categorical([0.2, 0.3, 0.5]),
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


def test_categorical_and_alias_sampler_degenerate_probabilities():
    assert cimba.random.categorical([0.0, 1.0]) == 1

    sampler = cimba.random.AliasSampler([0.0, 1.0])
    assert len(sampler) == 2
    assert sampler.sample() == 1
    sampler.close()
    sampler.close()
    with pytest.raises(RuntimeError, match="closed"):
        sampler.sample()

    with cimba.random.AliasSampler([1.0, 0.0]) as scoped:
        assert scoped.sample() == 0
    with pytest.raises(RuntimeError, match="closed"):
        scoped.sample()


def test_discrete_distribution_p_one_boundaries():
    assert cimba.random.geometric(1.0) == 1
    assert cimba.random.binomial(3, 1.0) == 3
    assert cimba.random.negative_binomial(3, 1.0) == 0


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
        cimba.random.categorical(probabilities)
    with pytest.raises(ValueError):
        cimba.random.AliasSampler(probabilities)


def test_sequence_and_unsigned_validation():
    with pytest.raises(ValueError, match="same length"):
        cimba.random.hyperexponential([1.0, 2.0], [1.0])
    with pytest.raises(OverflowError):
        cimba.random.erlang(1 << 32, 1.0)
    with pytest.raises(OverflowError):
        cimba.random.binomial(1 << 32, 0.5)
