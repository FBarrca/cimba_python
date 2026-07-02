from tutorial import tut_1_6


def test_tut_1_6_serial_experiment_sweep_preserves_utilization_trend():
    rhos, values = tut_1_6.sweep_rho(
        replications=1,
        duration=2500.0,
        warmup=100.0,
        seed=16,
    )

    assert rhos[0] < rhos[-1]
    assert values[0, 0] < values[-1, 0]
