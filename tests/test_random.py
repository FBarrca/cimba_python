import math

import cimba


def test_random_generator_is_repeatable_and_distributions_are_in_range():
    cimba.seed(0xC0FFEE)
    first = [cimba.random_u64() for _ in range(5)]
    cimba.seed(0xC0FFEE)
    second = [cimba.random_u64() for _ in range(5)]
    assert first == second

    assert 1.0 <= cimba.uniform(1.0, 2.0) <= 2.0
    assert 0.0 <= cimba.triangular(0.0, 0.5, 1.0) <= 1.0
    assert 0.0 <= cimba.pert(0.0, 0.5, 1.0) <= 1.0
    assert 0.0 <= cimba.beta(2.0, 3.0) <= 1.0
    assert cimba.exponential(1.0) >= 0.0
    assert cimba.gamma(2.0) >= 0.0
    assert 1 <= cimba.dice(1, 6) <= 6
    assert isinstance(cimba.flip(), bool)
    assert isinstance(cimba.bernoulli(0.25), bool)
    assert math.isfinite(cimba.normal())
