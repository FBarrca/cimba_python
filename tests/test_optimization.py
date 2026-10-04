"""Domains, batch execution and the optimization statistical contract."""

from dataclasses import FrozenInstanceError
from math import log
import os

import numpy as np
import pytest

import cimba as cb
from cimba.experiments import ExperimentConfigError, TrialsFailed, trial_seed
from cimba.optimize import DifferentialEvolution, OptimizationError
from cimba.optimize.space import Space
from cimba.schema import Assembly, ModelDefinitionError


class Quadratic(cb.Model):
    x: cb.Param[float] = 1.0
    target: cb.Param[float] = 2.0
    noise: cb.Input[float] = cb.inputs.dist.normal(mean=0, sd=1)
    cost: cb.Output[float]
    observations: cb.Dataset

    @cb.process
    def measure(self):
        self.cost = (self.x - self.target) ** 2 + self.noise.next()
        self.observations.record(self.cost)


def study(*, seed=11, replications=8):
    model = Quadratic()
    model.x = decision = cb.decision(-5, 5, step=1)
    optimization = cb.Optimization(model, minimize=lambda r: r[model].cost,
                                   replications=replications, seed=seed)
    return model, decision, optimization


def test_reproduction_cache_snapshot_apply_and_immutable_answer(monkeypatch):
    model, decision, optimization = study()
    model.observations.capture()
    values = {decision: 2.0}
    calls = []
    original = optimization.snapshot.run

    def run(*args, **kwargs):
        calls.append(kwargs.get('captures'))
        return original(*args, **kwargs)

    monkeypatch.setattr(optimization.snapshot, 'run', run)
    evaluated = optimization.evaluate([values, values])
    assert evaluated[0] is evaluated[1]
    assert calls == [False]
    assert optimization.evaluate([values])[0] is evaluated[0]
    assert calls == [False]
    replay = optimization.experiment(values)
    assert model.x is decision
    np.testing.assert_array_equal(evaluated[0].samples, replay.run()[model].cost[0])
    model.target = 99  # study and its reproduction use the saved configuration
    replay = optimization.experiment(values)
    assert model.target == 99
    np.testing.assert_array_equal(evaluated[0].samples, replay.run()[model].cost[0])
    best = optimization.run(optimizer=DifferentialEvolution(popsize=5, maxiter=8),
                            initial=[values], validation=20)
    assert best[decision] == 2
    assert best.estimate.n == 20
    assert calls[-2:] == [True, True]
    assert best.batches[-2].phase == 'selection'
    assert best.batches[-1].phase == 'estimation'
    assert best.meta.stop_reason in {'converged', 'maxiter'}
    assert best.history.to_table()
    assert best.batches.to_table()
    with pytest.raises(TypeError):
        best.values[decision] = 1
    with pytest.raises(FrozenInstanceError):
        best.estimate = None
    with pytest.raises(ValueError):
        evaluated[0].samples[0] = 100
    model.target = 2
    best.apply()
    assert model.x == 2
    reproduced = cb.Experiment(model, replications=20, seed=best.meta.estimation_seed).run()
    np.testing.assert_array_equal(best.results[model].cost.values, reproduced[model].cost.values)
    assert best.results[model].observations.trial(0, 0)[1].size == 1
    search = cb.Experiment(model, replications=8, seed=11).run()
    np.testing.assert_array_equal(search[model].cost[0], evaluated[0].samples)
    seed_sets = [set(trial_seed(seed, r) for r in range(20))
                 for seed in (11, best.meta.selection_seed, best.meta.estimation_seed)]
    assert all(not seed_sets[a] & seed_sets[b] for a, b in ((0, 1), (0, 2), (1, 2)))


def test_batch_calls_budget_recompilation_and_progress(monkeypatch):
    import cimba.experiments.snapshot as runner
    model, decision, optimization = study()
    optimization.evaluate([{decision: 0}])  # compile before counting misses
    misses = cb.cache_info().misses
    calls = []
    original = runner.run_blocks

    def run(blocks, **kwargs):
        calls.append((len(blocks), kwargs['workers']))
        return original(blocks, **kwargs)

    monkeypatch.setattr(runner, 'run_blocks', run)
    best = optimization.run(evaluations=8, optimizer=DifferentialEvolution(popsize=5, tol=0),
                            initial=[{decision: 2}], validation=6, workers=4)
    assert best.meta.evaluations <= 8
    assert best.meta.stop_reason == 'budget'
    assert cb.cache_info().misses == misses
    assert len(calls) == len(best.batches)
    assert all(workers == 4 for _, workers in calls)
    assert [count for count, _ in calls] == [batch.trials for batch in best.batches]
    search = [batch for batch in best.batches if batch.phase == 'search']
    assert sum(batch.simulated for batch in search) == best.meta.evaluations
    assert best.meta.trials.search == sum(batch.trials for batch in search)
    assert any(row.cached for row in best.history)
    updates = []
    stopped = optimization.run(optimizer=DifferentialEvolution(popsize=5), validation=6,
                               progress=lambda update: updates.append(update) or True)
    assert stopped.meta.stop_reason == 'progress'
    assert len(updates) == 1
    initial_only = study()[2].run(evaluations=5, optimizer=DifferentialEvolution(popsize=5), validation=6)
    assert initial_only.meta.stop_reason == 'budget'
    assert initial_only.meta.generations == 0


def test_worker_rng_and_repeated_run_invariance():
    np.random.seed(1234)
    state = np.random.get_state()
    model, decision, optimization = study()
    settings = dict(optimizer=DifferentialEvolution(popsize=5, maxiter=5), validation=12)
    one = optimization.run(workers=1, **settings)
    many = optimization.run(workers=4, **settings)
    fresh = study()[2].run(workers=4, **settings)
    assert one[decision] == many[decision]
    assert one.estimate == many.estimate == fresh.estimate
    assert [tuple(r.values.values()) for r in one.history] == [tuple(r.values.values()) for r in fresh.history]
    assert [r.mean for r in one.history] == [r.mean for r in many.history]
    assert many.meta.evaluations == 0
    new_state = np.random.get_state()
    assert state[0] == new_state[0]
    np.testing.assert_array_equal(state[1], new_state[1])
    assert state[2:] == new_state[2:]
    changed = study(seed=12)[2].run(**settings)
    assert [tuple(r.values.values()) for r in fresh.history] != [tuple(r.values.values()) for r in changed.history]


class Domains(cb.Model):
    continuous: cb.Param[float] = 1
    lattice: cb.Param[float] = 1
    logarithmic: cb.Param[float] = 1
    count: cb.Param[int] = 1
    switch: cb.Param[bool] = False
    linked: cb.Param[float] = 1
    derived: cb.Param[float] = 1
    state: cb.State[float] = 1


def test_domains_linking_mapping_levels_and_description():
    m = Domains()
    m.continuous = continuous = cb.decision(1, 9)
    m.linked = continuous
    m.derived = continuous.map(lambda x: x / 2).map(lambda x: x + 1)
    m.lattice = lattice = cb.decision(10, 20, step=2)
    m.logarithmic = logarithmic = cb.decision(1, 100, log=True)
    m.count = count = cb.decision(1, 5)
    m.switch = switch = cb.decision(False, True)
    space = Space(Assembly.of(m))
    assert len(space.dimensions) == 5
    assert space.integrality == (False, True, False, True, True)
    candidate = space.decode([[4, 2.6, log(10), 3.6, 0.8]])[0]
    design = space.design([candidate])
    assert design.points[0].bindings[m, 'count'] == 4
    assert design.points[0].bindings[m, 'switch'] is True
    assert design.points[0].bindings[m, 'linked'] == 4
    assert design.points[0].bindings[m, 'derived'] == 3
    assert design.levels(m.derived) == (3,)
    assert space.values(candidate)[lattice] == 16
    assert space.values(candidate)[logarithmic] == pytest.approx(10)
    with pytest.raises(ValueError, match='every decision'):
        space.encode({continuous: 2})
    values = {continuous: 3, lattice: 14, logarithmic: 10, count: 3, switch: False}
    np.testing.assert_allclose(space.encode(values), [3, 2, log(10), 3, 0])
    with pytest.raises(ValueError, match='lattice'):
        space.encode(values | {lattice: 13})
    with pytest.raises(ModelDefinitionError, match='continuous.*decision has no value'):
        cb.Experiment(m)
    with pytest.raises(TypeError, match='Param'):
        m.state = continuous


@pytest.mark.parametrize('bounds,kwargs', [((1, 1), {}), ((2, 1), {}), ((0, 1), {'step': 0}),
    ((0, 1), {'step': .3}), ((0, 1), {'log': True}), ((1, 3), {'step': 1, 'log': True}),
    ((0, float('inf')), {}), ((0, 1), {'step': float('nan')})])
def test_invalid_decision_arguments(bounds, kwargs):
    with pytest.raises(ValueError):
        cb.decision(*bounds, **kwargs)


@pytest.mark.parametrize('kwargs', [{'popsize': 0}, {'popsize': 1.2}, {'mutation': 2}, {'mutation': (-1, 1)},
    {'mutation': (1, .5)}, {'recombination': 1.1}, {'strategy': 'unknown'}, {'maxiter': -1},
    {'tol': float('nan')}, {'init': 'unknown'}])
def test_invalid_de_arguments(kwargs):
    with pytest.raises(ValueError):
        DifferentialEvolution(**kwargs)


def test_model_and_run_configuration_errors():
    with pytest.raises(TypeError):
        DifferentialEvolution(workers=2)
    with pytest.raises(ModelDefinitionError, match='nothing to optimize'):
        cb.Optimization(Quadratic(), minimize=lambda r: 1)
    m = Domains()
    m.count = cb.decision(1.5, 5)
    with pytest.raises(ModelDefinitionError, match='count.*integral'):
        cb.Optimization(m, minimize=lambda r: 1)
    m.count = cb.decision(1, 5, log=True)
    with pytest.raises(ModelDefinitionError, match='count.*log'):
        cb.Optimization(m, minimize=lambda r: 1)
    m.count = m.continuous = cb.decision(1, 5)
    with pytest.raises(ModelDefinitionError, match='count.*different types'):
        cb.Optimization(m, minimize=lambda r: 1)
    m.count = cb.sweep(1.5, 2.7)
    m.continuous = 2
    with pytest.raises(ModelDefinitionError, match='count.*integral'):
        cb.Experiment(m)
    m.continuous = cb.decision(1, 5)
    with pytest.raises(ModelDefinitionError, match='count.*sweeps'):
        cb.Optimization(m, minimize=lambda r: 1)
    model, d, optimization = study()
    for kwargs in ({'evaluations': 1}, {'finalists': 0}, {'validation': 1}, {'on_failure': 'record'}, {'workers': 0}):
        with pytest.raises(ExperimentConfigError):
            optimization.run(**kwargs)
    with pytest.raises(ValueError, match='outside'):
        optimization.run(initial=[{d: 99}])
    with pytest.raises(ValueError, match='every decision'):
        optimization.run(initial=[{}])


@pytest.mark.parametrize('objective', [lambda r: 1., lambda r: np.zeros(5), lambda r: [['text']], lambda r: np.zeros((3, 2))])
def test_invalid_objectives_name_generation(objective):
    model = Quadratic()
    model.x = cb.decision(-5, 5)
    with pytest.raises(OptimizationError, match='generation 0'):
        cb.Optimization(model, minimize=objective, replications=4).run(
            optimizer=DifferentialEvolution(popsize=5), validation=4)


def test_failure_policies_and_nonfinite_objectives():
    model = Quadratic()
    model.x = d = cb.decision(-5, 5)

    def objective(results):
        values = results[model].cost.values.copy()
        values[:, 0] = np.nan
        return values

    optimization = cb.Optimization(model, minimize=objective, replications=4)
    record = optimization.evaluate([{d: 2}])[0]
    assert record.estimate.n == 3
    assert record.reasons == ('objective is not finite',)
    assert record.energy(1, 'reject') == float('inf')
    assert np.isfinite(record.energy(1, 'ignore'))
    with pytest.raises(TrialsFailed, match='candidate.*not finite'):
        optimization.evaluate([{d: 2}], on_failure='raise')
    with pytest.raises(OptimizationError, match='initial population.*not finite'):
        optimization.run(optimizer=DifferentialEvolution(popsize=5), validation=4)
    best = optimization.run(on_failure='ignore', optimizer=DifferentialEvolution(popsize=5, maxiter=2), validation=4)
    assert best.estimate.n == 3


@pytest.mark.parametrize('phase', ['selection', 'estimation'])
@pytest.mark.parametrize('on_failure,error', [('reject', OptimizationError), ('raise', TrialsFailed)])
def test_independent_validation_failures_cannot_produce_an_answer(phase, on_failure, error):
    model = Quadratic()
    model.x = cb.decision(-5, 5, step=1)

    def objective(results):
        values = results[model].cost.values.copy()
        seed = getattr(optimization, f'{phase}_seed')
        if results.seeds[0, 0] == trial_seed(seed, 0):
            values[:, 0] = np.nan
        return values

    optimization = cb.Optimization(model, minimize=objective, replications=4)
    with pytest.raises(error, match=phase):
        optimization.run(on_failure=on_failure,
                         optimizer=DifferentialEvolution(popsize=5, maxiter=0), validation=4)


def test_maximize_preserves_objective_units_and_known_optimum():
    model = Quadratic()
    model.x = d = cb.decision(-5, 5, step=1)
    best = cb.Optimization(model, maximize=lambda r: -r[model].cost, replications=8).run(
        optimizer=DifferentialEvolution(popsize=10, maxiter=20), initial=[{d: 2}], validation=20)
    assert best[d] == 2
    assert best.estimate.mean == pytest.approx(np.mean(-best.results[model].cost.values))


class Newsvendor(cb.Model):
    quantity: cb.Param[float] = 1
    demand: cb.Input[float] = cb.inputs.dist.exponential(mean=10)
    cost: cb.Output[float]

    @cb.process
    def buy(self):
        demand = self.demand.next()
        self.cost = max(0., self.quantity - demand) + 4 * max(0., demand - self.quantity)


def test_newsvendor_analytic_optimum_and_paired_finalists():
    model = Newsvendor()
    model.quantity = d = cb.decision(1., 40.)
    best = cb.Optimization(model, minimize=lambda r: r[model].cost, replications=512, seed=4).run(
        optimizer=DifferentialEvolution(popsize=10, tol=1e-5, maxiter=20), validation=256)
    assert abs(best[d] - 10 * log(5)) < 2
    assert best.results.levels(d) == (best[d],)
    assert all(f.difference.n == 256 for f in best.finalists)
    assert any(f.difference.difference == 0 for f in best.finalists)


def test_fill_and_initial_population_modes():
    _, d, optimization = study(replications=8)
    for init in ('random', 'latinhypercube', 'sobol', 'halton'):
        best = optimization.run(optimizer=DifferentialEvolution(popsize='fill', init=init, maxiter=0),
                                initial=[{d: 2}], workers=1, validation=4)
        assert best[d] == 2
        assert best.meta.optimizer.popsize == 32
        assert best.history[0].values[d] == 2


@pytest.mark.parametrize('strategy', [name + kind for name in
    ('best1', 'best2', 'rand1', 'rand2', 'randtobest1', 'currenttobest1') for kind in ('bin', 'exp')])
def test_all_de_strategies_have_enough_population_members(strategy):
    model, decision, optimization = study()
    best = optimization.run(optimizer=DifferentialEvolution(popsize=1, strategy=strategy, maxiter=1),
                            validation=4)
    assert best.estimate.n == 4


def test_log_values_reproduce_without_rounding_and_mapping_precedes_source():
    model = Quadratic()
    model.x = d = cb.decision(.01, 100, log=True)
    optimization = cb.Optimization(model, minimize=lambda r: r[model].cost, replications=4)
    value = float(np.nextafter(7.3, 8.))
    record = optimization.evaluate([{d: value}])[0]
    assert record.values[d] == value
    replay = optimization.experiment({d: value}).run()
    np.testing.assert_array_equal(record.samples, replay[model].cost[0])
    mapped = Domains()
    root = cb.decision(1., 9.)
    mapped.continuous = root.map(lambda x: 2 * x)
    mapped.logarithmic = root
    space = Space(Assembly.of(mapped))
    assert len(space.dimensions) == 1
    assert space.values((3,))[mapped.continuous] == 6


def test_samples_arithmetic_ufuncs_and_readonly_output():
    samples = cb.Samples(np.arange(6., dtype=float).reshape(2, 3))
    np.testing.assert_array_equal(2 * samples + 1 - samples, samples.values + 1)
    np.testing.assert_array_equal(np.sqrt(samples), np.sqrt(samples.values))
    np.testing.assert_array_equal(samples >= 3, samples.values >= 3)
    np.testing.assert_array_equal(np.add.reduce(samples, axis=1), [3, 12])
    destination = np.zeros((2, 3))
    np.add(samples, 1, out=destination)
    np.testing.assert_array_equal(destination, samples.values + 1)
    with pytest.raises(ValueError):
        np.add(samples, 1, out=samples)
    with pytest.raises(ValueError):
        samples += 1


def test_generated_rows_are_cached_and_chunking_preserves_search():
    generated = []

    def generator(rng, length):
        generated.append(length)
        return rng.normal(size=length)

    model = Quadratic()
    model.noise = cb.inputs.row_source('optimization.rows', [1, 2, 3], generator,
                                      length_hint=2, on_exhausted='fail')
    model.x = d = cb.decision(-5, 5, step=1)
    optimization = cb.Optimization(model, minimize=lambda r: r[model].cost, replications=8, seed=3)
    optimization.evaluate([{d: 0}])
    assert len(generated) == 8
    optimization.evaluate([{d: 1}, {d: 2}])
    assert len(generated) == 8
    settings = dict(optimizer=DifferentialEvolution(popsize=5, maxiter=4), validation=8)
    full = optimization.run(**settings)
    small = optimization.run(input_memory=32, **settings)
    assert full[d] == small[d]
    assert full.estimate == small.estimate
    assert [r.mean for r in full.history] == [r.mean for r in small.history]
    assert small.results.meta.chunks == 4
    assert optimization.snapshot._row_bytes <= 32
    new = cb.Optimization(model, minimize=lambda r: r[model].cost, replications=8, seed=3)
    chunked = new.run(input_memory=32, workers=4, **settings)
    assert chunked[d] == full[d]
    assert chunked.estimate == full.estimate
    assert [row.mean for row in chunked.history] == [row.mean for row in full.history]
    assert len(chunked.batches) == sum(row.native_calls for row in chunked.batches)


class FailedTrials(cb.Model):
    count: cb.Param[int] = 1
    source: cb.Input[float] = cb.inputs.trace([1])
    cost: cb.Output[float]

    @cb.process
    def consume(self):
        for _ in range(self.count):
            self.source.next()
        self.cost = -self.count


def test_native_failures_are_rejected_with_candidate_reasons():
    model = FailedTrials()
    model.count = d = cb.decision(1, 3)
    optimization = cb.Optimization(model, minimize=lambda r: r[model].cost, replications=4)
    records = optimization.evaluate([{d: 1}, {d: 2}])
    assert records[0].estimate.mean == -1
    assert records[1].failed.all()
    assert 'source exhausted' in records[1].reasons[0]
    best = optimization.run(optimizer=DifferentialEvolution(popsize=5, maxiter=3),
                            initial=[{d: 1}], validation=4)
    assert best[d] == 1
    assert any(row.failed == 4 for row in best.history)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('CIMBA_SLOW_TESTS') != '1', reason='opt-in statistical coverage check')
def test_independent_estimate_covers_analytic_cost_at_selected_quantity():
    from scipy.stats import binomtest
    covered = 0
    repetitions = 40
    for seed in range(repetitions):
        model = Newsvendor()
        model.quantity = d = cb.decision(1., 40.)
        best = cb.Optimization(model, minimize=lambda r: r[model].cost,
                               replications=32, seed=seed).run(
            optimizer=DifferentialEvolution(popsize=5, maxiter=5), validation=256)
        quantity = best[d]
        true_cost = quantity - 10 + 50 * np.exp(-quantity / 10)
        covered += best.estimate.lower <= true_cost <= best.estimate.upper
    assert binomtest(covered, repetitions, .95).pvalue > .01


@pytest.mark.benchmark
@pytest.mark.skipif(os.environ.get('CIMBA_BENCHMARK') != '1', reason='opt-in hardware benchmark')
def test_tutorial_store_host_work_is_under_quarter_of_native_time():
    from tutorial.policy_comparison import MinMax, Store
    policy = MinMax()
    policy.minimum = minimum = cb.decision(0., 100.)
    policy.maximum = maximum = cb.decision(40., 250.)
    store = Store()
    store.policy = policy

    def objective(results):
        s = results[store]
        return s.mean_stock + 20 * s.orders_per_week + 400 * (1 - s.fill_rate)

    optimization = cb.Optimization(store, minimize=objective, replications=32,
                                   window=cb.Window(warmup=30, duration=365), seed=11)
    optimization.evaluate([{minimum: 40, maximum: 120}], workers=1)  # warm compilation
    rng = np.random.default_rng(4)
    ratios = []
    for generation in range(3):
        candidates = tuple(tuple(values) for values in rng.uniform([0, 40], [100, 250], (120, 2)))
        _, _, batches = optimization._evaluate(candidates, generation=generation,
                                               workers=None, input_memory=1 << 30, on_failure='reject')
        assert len(batches) == 1
        batch = batches[0]
        assert batch.trials == 3840
        ratios.append(batch.host_wall_time / batch.native_wall_time)
    assert min(ratios) <= .25, ratios
    best = optimization.run(optimizer=DifferentialEvolution(popsize=60, maxiter=0), validation=8)
    if (os.cpu_count() or 1) > 1:
        assert best.batches[0].busy_cpus > 1
