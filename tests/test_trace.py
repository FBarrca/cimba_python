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


def test_trace_view_requires_compiled_code():
    with pytest.raises(TypeError, match="compiled model code"):
        sim.Trace(np.zeros(2, dtype=np.int64))
