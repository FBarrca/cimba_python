import numpy as np
import pytest

from cimba import analysis, inputs
from cimba.results import Samples


def test_summary_masks_failures_and_compare_is_paired():
    samples = Samples(np.array([[1.0, 2.0, np.nan],
                                [2.0, 4.0, 10.0]]))
    readout = analysis.summary(samples)
    np.testing.assert_array_equal(readout.n, [2, 3])
    assert readout.mean[0] == 1.5
    comparison = analysis.compare(samples, a=0, b=1)
    assert comparison.n == 2
    assert comparison.difference == 1.5


def test_check_input_and_fit_expose_input_model_quality():
    rng = np.random.default_rng(17)
    data = rng.exponential(2.0, size=1000)
    fitted = inputs.fit(data, [inputs.dist.exponential, inputs.dist.normal])
    assert fitted[0].source.method == "exponential"
    report = analysis.check_input(inputs.bootstrap.iid(data),
                                  reference=data, n=1000, seed=10)
    assert abs(report.generated_mean - report.reference_mean) < 0.2
    assert len(report.generated_acf) == 11
    assert len(report.generated_pacf) == 11


@pytest.mark.parametrize("source", [
    inputs.dist.erlang(k=2, mean=3),
    inputs.dist.beta(a=2, b=3),
    inputs.dist.negative_binomial(successes=3, p=0.5),
    inputs.dist.hyperexponential([1, 3], [0.5, 0.5]),
    inputs.dist.hypoexponential([1, 3]),
])
def test_check_input_covers_distribution_catalogue(source):
    reference = np.random.default_rng(11).normal(size=256)
    report = analysis.check_input(source, reference=reference, n=256)
    assert np.isfinite(report.generated_mean)
    assert len(report.generated_pacf) == 11


def test_check_joint_input_preserves_cross_correlation():
    a = np.arange(1.0, 101.0)
    panel = {"a": a, "b": 3.0 * a}
    joint = inputs.bootstrap.joint(panel, mean_block=5)
    report = analysis.check_input(joint, reference=panel, n=300, seed=14)
    assert report.keys == ("a", "b")
    assert len(report.members) == 2
    np.testing.assert_allclose(report.reference_correlation[0, 1], 1.0)
    np.testing.assert_allclose(report.generated_correlation[0, 1], 1.0)
    assert not report.generated_correlation.flags.writeable
