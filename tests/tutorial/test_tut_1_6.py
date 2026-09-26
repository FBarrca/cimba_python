from tutorial.tut_1_6 import sweep_rho


def test_utilization_sweep_preserves_trend():
    rhos, values = sweep_rho(replications=1, duration=2_500,
                             warmup=100, seed=16)
    assert values.shape == (len(rhos), 1)
    assert values[0, 0] < values[-1, 0]
