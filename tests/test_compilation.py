"""Compilation belongs to a fully constructed, concrete model."""
import types

import numpy as np
import pytest

import cimba.sim as sim


_configuration = types.ModuleType("compilation_test_config")
_configuration.VALUE = 2.0


class Configured(sim.Model):
    result: sim.Output
    helper_result: sim.Output

    @sim.function
    def read(self) -> float:
        return _configuration.VALUE

    @sim.process
    def run(self):
        self.result = _configuration.VALUE
        self.helper_result = self.read()


def test_new_models_observe_configuration_and_compiled_models_reuse_it(monkeypatch):
    monkeypatch.setattr(_configuration, "VALUE", 2.0)
    first = Configured()
    assert first._compiled is None
    assert first.compile() is first
    compiled = first._compiled
    monkeypatch.setattr(_configuration, "VALUE", 9.0)
    second = Configured()
    for model, expected in [(first, 2.0), (second, 9.0), (first, 2.0)]:
        experiment = model.experiment(duration=None, warmup=0, seed=1)
        assert experiment.run() == 0
        np.testing.assert_array_equal(experiment["result"], expected)
        np.testing.assert_array_equal(experiment["helper_result"], expected)
    assert first._compiled is compiled
    assert second._compiled is not compiled


def test_compilation_occurs_after_subclass_initialization(monkeypatch):
    calls = []
    original = sim.Model._compile_callbacks

    def check(model, *args, **kwargs):
        assert model.initialized
        calls.append(model)
        return original(model, *args, **kwargs)

    monkeypatch.setattr(sim.Model, "_compile_callbacks", check)

    class Complete(Configured):
        def __init__(self):
            super().__init__()
            self.initialized = True

    model = Complete()
    assert not calls
    model.compile()
    model.compile()
    model.experiment(duration=None, warmup=0)
    assert calls == [model]


def test_errors_are_raised_at_compile_and_new_model_can_retry(monkeypatch):
    monkeypatch.setattr(_configuration, "VALUE", "bad")
    model = Configured()
    with pytest.raises(TypeError, match="failed Numba nopython compilation"):
        model.compile()
    assert model._compiled is None
    monkeypatch.setattr(_configuration, "VALUE", 4.0)
    experiment = Configured().experiment(duration=None, warmup=0)
    assert experiment.run() == 0
    assert experiment["helper_result"][0] == 4.0


def test_removed_class_compilation_mode_has_migration_diagnostic():
    with pytest.raises(TypeError, match=r"call model.compile\(\)"):
        class Obsolete(Configured):
            __cimba_precompile__ = "eager"


def test_different_record_layouts_do_not_share_user_callbacks():
    class Cell(sim.Component):
        total: sim.Output
        stream: sim.Trace

        @sim.process
        def consume(self, env):
            self.total = sim.Trace(self.stream).sum()

    class Network(sim.Model):
        cells: list[Cell] = [Cell(), Cell()]

    for model in [Network(), Network(state=["extra"]), Network()]:
        experiment = model.experiment(
            cells__stream=np.array([[1., 2.], [3., 4.]]),
            duration=None, warmup=0)
        assert experiment.run() == 0
        np.testing.assert_array_equal(experiment["cells__total"], [[3., 7.]])


def test_native_callback_symbols_do_not_collide_between_models(monkeypatch):
    class Direct(sim.Model):
        result: sim.Output

        @sim.process
        def run(self):
            self.result = _configuration.VALUE

    # No synchronous helpers compile in the parent to advance Numba's symbol
    # counter. Forked workers must still produce distinct native entry points.
    for value in [1.0, 8.0, 3.0]:
        monkeypatch.setattr(_configuration, "VALUE", value)
        experiment = Direct().experiment(duration=None, warmup=0)
        assert experiment.run() == 0
        np.testing.assert_array_equal(experiment["result"], value)


def test_concurrent_experiments_share_one_compilation(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    original = sim.Model._compile_callbacks
    calls = []

    def observe(model, *args, **kwargs):
        calls.append(model)
        return original(model, *args, **kwargs)

    monkeypatch.setattr(sim.Model, "_compile_callbacks", observe)
    model = Configured()

    def experiment(_):
        return model.experiment(duration=None, warmup=0)

    with ThreadPoolExecutor(max_workers=3) as pool:
        experiments = list(pool.map(experiment, range(3)))
    assert calls == [model]
    for item in experiments:
        assert item.run() == 0
