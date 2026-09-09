import math
import inspect

import numpy as np

import pytest

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


@pytest.mark.parametrize("name", cimba.random.__all__)
def test_random_draws_require_compiled_model_code(name):
    draw = getattr(cimba.random, name)
    arguments = {name: 1 for name in inspect.signature(draw).parameters}
    with pytest.raises(RuntimeError, match="compiled model callback"):
        draw(**arguments)


class Categorical(sim.Model):
    probabilities: sim.Trace
    value: sim.Output

    @sim.process
    def draw(self):
        self.value = cimba.random.categorical(sim.Trace(self.probabilities))


@pytest.fixture(scope="module")
def categorical_model():
    return Categorical().compile()


@pytest.mark.parametrize("probabilities", [[], [math.nan], [math.inf], [-0.1, 1.1], [0.2, 0.2]])
def test_invalid_probabilities_fail_model_trials(categorical_model, probabilities):
    exp = categorical_model.experiment(
        probabilities=np.array(probabilities, dtype=np.float64),
        replications=4, duration=1.0, warmup=0.0, seed=5)
    assert exp.run() == 4
    assert exp.failed.all()
    assert np.isnan(exp["value"]).all()


@pytest.mark.parametrize("probabilities, expected", [([0.0, 1.0], 1), ([1.0, 0.0], 0)])
def test_categorical_degenerate_probabilities(categorical_model, probabilities, expected):
    exp = categorical_model.experiment(
        probabilities=np.array(probabilities),
        replications=8, duration=1.0, warmup=0.0, seed=5)
    assert exp.run() == 0
    assert (exp["value"] == expected).all()
