from tutorial import tut_1_7


def test_tut_1_7_sweep_accepts_command_line_parameter_values():
    rhos, values = tut_1_7.sweep_rho(
        replications=1,
        duration=1000.0,
        warmup=10.0,
        seed=17,
    )

    assert rhos.shape == (39,)
    assert values.shape == (39, 1)
    assert values[0, 0] >= 0.0
