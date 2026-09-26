"""Every public parametric source runs through the same compiled input slot."""

import numpy as np
import pytest

import cimba as cb
from cimba import inputs


class Draw(cb.Model):
    value: cb.Input[float]
    sample: cb.Output[float]

    @cb.process
    def draw(self):
        self.sample = self.value.next()


def test_distribution_catalogue_uses_one_model_class():
    dist = inputs.dist
    sources = [
        dist.exponential(mean=2), dist.normal(mean=3, sd=2),
        dist.gamma(shape=2, scale=3), dist.lognormal(mean=0, sigma=1),
        dist.weibull(shape=2, scale=3), dist.poisson(mean=4),
        dist.triangular(low=0, mode=1, high=3),
        dist.pert(low=0, mode=1, high=3),
        dist.categorical([2, 4], [0.25, 0.75]),
        dist.uniform(low=1, high=3), dist.logistic(mean=0, scale=1),
        dist.cauchy(mode=0, scale=1), dist.erlang(k=2, mean=3),
        dist.beta(a=2, b=3), dist.pert_mod(low=0, mode=1, high=3, weight=4),
        dist.rayleigh(scale=2), dist.bernoulli(p=0.4),
        dist.geometric(p=0.5), dist.binomial(n=5, p=0.4),
        dist.negative_binomial(successes=3, p=0.5),
        dist.pareto(shape=3, mode=2), dist.chi_squared(df=3),
        dist.f_dist(df1=4, df2=6), dist.student_t(df=4),
        dist.dice(low=1, high=6),
        dist.hyperexponential([1, 3], [0.5, 0.5]),
        dist.hypoexponential([1, 3]),
    ]
    for source in sources:
        model = Draw()
        model.value = source
        result = cb.Experiment(model, replications=8, seed=71).run(workers=1)
        assert not result.failed.any(), source.method
        assert np.isfinite(result[model].sample.values).all(), source.method
        assert result[model].value.source[0]["method"] == f"dist.{source.method}"
        assert result.meta.compile.misses in (0, 1)


def test_invalid_distribution_sweep_is_rejected_before_native_run():
    model = Draw()
    model.value = inputs.dist.gamma(shape=cb.sweep(2.0, -1.0), scale=1.0)
    with pytest.raises(ValueError, match="gamma"):
        cb.Experiment(model)


class MeanDraw(cb.Model):
    value: cb.Input[float]
    mean: cb.Output[float]

    @cb.process
    def draw(self):
        total = 0.0
        for _ in range(20_000):
            total += self.value.next()
        self.mean = total / 20_000


@pytest.mark.parametrize("source, expected", [
    (inputs.dist.binomial(n=5, p=0.4), 2.0),
    (inputs.dist.negative_binomial(successes=3, p=0.5), 3.0),
    (inputs.dist.geometric(p=0.5), 2.0),
    (inputs.dist.hyperexponential([1.0, 3.0], [0.5, 0.5]), 2.0),
    (inputs.dist.hypoexponential([1.0, 3.0]), 4.0),
])
def test_distribution_means(source, expected):
    model = MeanDraw()
    model.value = source
    observed = cb.Experiment(model, seed=98).run(workers=1)[model].mean.values[0, 0]
    assert abs(observed - expected) < 0.15


def test_only_reruns_selected_trials_with_original_seeds():
    model = Draw()
    model.value = inputs.dist.normal(mean=2.0, sd=1.0)
    experiment = cb.Experiment(model, replications=6, seed=310)
    full = experiment.run(workers=2)
    focused = experiment.only(trials=[4, 1], workers=1)
    assert focused[model].sample.values[:, 0].tolist() == [
        full[model].sample.values[0, 4],
        full[model].sample.values[0, 1],
    ]
    assert focused.meta.compile.misses == 0
