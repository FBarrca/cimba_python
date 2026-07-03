"""Data-driven trace replay: sim.Trace declarations and experiment() data."""

import numpy as np
import pytest

import cimba.sim as sim


class Replay(sim.Model):
    scale: sim.Param
    demand: sim.Trace
    total: sim.Output
    length: sim.Output
    last: sim.Output
    exhausted_at: sim.Output


@pytest.fixture(scope="module")
def replay_model():
    model = Replay()

    @model.process
    def generator(env: Replay):
        demand = sim.Trace(env.demand)
        total = 0.0
        for x in demand:
            sim.hold(1.0)
            total += x * env.scale
        env.total = total
        env.length = float(len(demand))
        env.last = demand[len(demand) - 1]
        env.exhausted_at = sim.now()

    return model


def run(model, **kwargs):
    kwargs.setdefault("duration", 100.0)
    kwargs.setdefault("warmup", 0.0)
    exp = model.experiment(**kwargs)
    assert exp.run() == 0
    return exp


def test_shared_1d_trace_replays_in_every_trial(replay_model):
    trace = np.array([1.0, 2.0, 3.5])
    exp = run(replay_model, scale=1.0, demand=trace,
              replications=3, seed=1)
    assert np.allclose(exp["total"], 6.5)
    assert np.allclose(exp["length"], 3.0)
    assert np.allclose(exp["last"], 3.5)
    # The generator consumed one entry per time unit and then finished
    assert np.allclose(exp["exhausted_at"], 3.0)


def test_2d_trace_rows_align_with_trial_order(replay_model):
    reps = 3
    rows = np.arange(2 * reps * 4, dtype=np.float64).reshape(2 * reps, 4)
    exp = run(replay_model, scale=[1.0, 2.0], demand=rows,
              replications=reps, seed=2)
    # Row i must land in trial i whatever the param sweep ordering is
    assert np.allclose(exp["total"], rows.sum(axis=1) * exp["scale"])
    assert np.allclose(exp["last"], rows[:, -1])


def test_ragged_traces_carry_per_trial_lengths(replay_model):
    rows = [np.array([1.0]),
            np.array([2.0, 3.0]),
            np.array([4.0, 5.0, 6.0])]
    exp = run(replay_model, scale=1.0, demand=rows,
              replications=3, seed=3)
    assert np.allclose(exp["length"], [1.0, 2.0, 3.0])
    assert np.allclose(exp["total"], [1.0, 5.0, 15.0])
    assert np.allclose(exp["last"], [1.0, 3.0, 6.0])
    assert np.allclose(exp["exhausted_at"], [1.0, 2.0, 3.0])


def test_flat_list_is_a_shared_trace(replay_model):
    exp = run(replay_model, scale=1.0, demand=[1.0, 2.0],
              replications=2, seed=4)
    assert np.allclose(exp["total"], 3.0)


def test_trace_arrays_are_kept_alive_by_the_experiment(replay_model):
    import gc
    exp = replay_model.experiment(scale=1.0,
                                  demand=np.linspace(0.0, 1.0, 5).copy(),
                                  replications=2, duration=100.0,
                                  warmup=0.0, seed=5)
    gc.collect()  # the only reference left is the experiment's keep-alive
    assert exp.run() == 0
    assert np.allclose(exp["total"], np.linspace(0.0, 1.0, 5).sum())


def test_missing_trace_raises(replay_model):
    with pytest.raises(ValueError, match="missing trace"):
        replay_model.experiment(scale=1.0)


def test_unknown_name_still_raises(replay_model):
    with pytest.raises(ValueError, match="unknown"):
        replay_model.experiment(scale=1.0, demand=np.zeros(1), bogus=1.0)


def test_2d_row_count_must_match_trials(replay_model):
    with pytest.raises(ValueError, match="expected 3 rows"):
        replay_model.experiment(scale=1.0, replications=3,
                                demand=np.zeros((2, 4)))


def test_ragged_count_must_match_trials(replay_model):
    with pytest.raises(ValueError, match="per-trial arrays"):
        replay_model.experiment(scale=1.0, replications=3,
                                demand=[np.zeros(1), np.zeros(2)])


def test_3d_trace_rejected(replay_model):
    with pytest.raises(ValueError, match="1-D array"):
        replay_model.experiment(scale=1.0, demand=np.zeros((2, 2, 2)))


def test_callable_trace_reproduces_from_the_experiment_seed(replay_model):
    def bootstrap(rng):
        return rng.normal(10.0, 2.0, size=6)

    a = run(replay_model, scale=1.0, demand=bootstrap,
            replications=4, seed=11)
    b = run(replay_model, scale=1.0, demand=bootstrap,
            replications=4, seed=11)
    c = run(replay_model, scale=1.0, demand=bootstrap,
            replications=4, seed=12)
    assert np.allclose(a["total"], b["total"])
    assert not np.allclose(a["total"], c["total"])
    # Each trial draws its own stream
    assert len(np.unique(a["total"])) == 4


def test_callable_trace_uses_the_trial_trace_rng(replay_model):
    def bootstrap(rng):
        return rng.normal(10.0, 2.0, size=6)

    exp = run(replay_model, scale=1.0, demand=bootstrap,
              replications=3, seed=21)
    for i in range(3):
        expected = bootstrap(sim.trace_rng(exp["seed"][i], "demand"))
        assert np.isclose(exp["total"][i], expected.sum())
        assert np.isclose(exp["last"][i], expected[-1])


def test_callable_trace_receives_the_trial_index(replay_model):
    def ramp(rng, trial):
        return np.full(2, float(trial))

    exp = run(replay_model, scale=1.0, demand=ramp,
              replications=3, seed=31)
    assert np.allclose(exp["total"], [0.0, 2.0, 4.0])


def test_trial_seeds_match_the_experiment_assignment(replay_model):
    seeds = replay_model.trial_seeds(seed=7, scale=[1.0, 2.0],
                                     replications=3)
    exp = run(replay_model, scale=[1.0, 2.0], demand=np.zeros(1),
              replications=3, seed=7)
    assert np.array_equal(seeds, exp["seed"])


def test_precomputed_rows_reproduce_the_callable_form(replay_model):
    def bootstrap(rng):
        return rng.normal(10.0, 2.0, size=6)

    seeds = replay_model.trial_seeds(seed=41, scale=1.0, replications=4)
    rows = [bootstrap(sim.trace_rng(s, "demand")) for s in seeds]
    pre = run(replay_model, scale=1.0, demand=rows,
              replications=4, seed=41)
    live = run(replay_model, scale=1.0, demand=bootstrap,
               replications=4, seed=41)
    assert np.array_equal(pre["total"], live["total"])
    assert np.array_equal(pre["last"], live["last"])


def test_trial_seeds_ignores_trace_kwargs_but_checks_params(replay_model):
    seeds = replay_model.trial_seeds(seed=7, scale=1.0, demand=np.zeros(1))
    assert seeds.shape == (1,)
    with pytest.raises(ValueError, match="missing parameter"):
        replay_model.trial_seeds(seed=7)
    with pytest.raises(ValueError, match="unknown"):
        replay_model.trial_seeds(seed=7, scale=1.0, bogus=1.0)


def test_callable_default_args_do_not_receive_the_trial_index(replay_model):
    def bootstrap(rng, size=4):
        return rng.normal(10.0, 2.0, size=size)

    exp = run(replay_model, scale=1.0, demand=bootstrap,
              replications=3, seed=51)
    # size must keep its default; only two required params opt in
    assert np.allclose(exp["length"], 4.0)


def test_bootstrap_factories_plug_into_experiment(replay_model):
    from cimba import bootstrap

    history = np.sin(np.arange(30.0))
    gen = bootstrap.stationary(history, length=8, mean_block=5)
    a = run(replay_model, scale=1.0, demand=gen, replications=3, seed=61)
    b = run(replay_model, scale=1.0, demand=gen, replications=3, seed=61)
    assert np.array_equal(a["total"], b["total"])
    assert np.allclose(a["length"], 8.0)
    assert len(np.unique(a["total"])) == 3  # per-trial resamples differ


def test_trace_rng_name_attribute_overrides_the_stream_tag(replay_model):
    def gen(rng):
        return rng.normal(10.0, 2.0, size=6)

    gen.trace_rng_name = "shared-demand"
    exp = run(replay_model, scale=1.0, demand=gen, replications=3, seed=71)
    for i in range(3):
        expected = gen(sim.trace_rng(exp["seed"][i], "shared-demand"))
        assert np.isclose(exp["total"][i], expected.sum())


def test_joint_traces_stay_correlated_through_experiment():
    from cimba import bootstrap

    class Pair(sim.Model):
        demand_a: sim.Trace
        demand_b: sim.Trace
        total_a: sim.Output
        total_b: sim.Output

    model = Pair()

    @model.process
    def consume_a(env: Pair):
        values = sim.Trace(env.demand_a)
        total = 0.0
        for x in values:
            sim.hold(1.0)
            total += x
        env.total_a = total

    @model.process
    def consume_b(env: Pair):
        values = sim.Trace(env.demand_b)
        total = 0.0
        for x in values:
            sim.hold(1.0)
            total += x
        env.total_b = total

    t = np.arange(60.0)
    hist_a = np.sin(t / 3.0)
    gens = bootstrap.joint({"demand_a": hist_a, "demand_b": -hist_a},
                           length=30, name="pair", mean_block=6)
    exp = model.experiment(**gens, replications=4, duration=100.0,
                           warmup=0.0, seed=81)
    assert exp.run() == 0
    # Identical per-trial block draws: the mirrored series stays mirrored
    assert np.allclose(exp["total_a"], -exp["total_b"])
    assert len(np.unique(exp["total_a"])) == 4  # trials still differ


def test_callable_trace_must_return_1d(replay_model):
    with pytest.raises(ValueError, match="must return a 1-D array"):
        replay_model.experiment(scale=1.0, replications=2,
                                demand=lambda rng: np.zeros((2, 2)))


def test_trace_view_requires_compiled_code():
    with pytest.raises(TypeError, match="compiled model code"):
        sim.Trace(np.zeros(2, dtype=np.int64))
