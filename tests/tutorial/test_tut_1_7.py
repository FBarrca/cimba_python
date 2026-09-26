from tutorial.tut_1_7 import sweep_rho


def test_command_line_sweep_shape():
    rhos, values = sweep_rho(replications=1, duration=1_000,
                             warmup=10, seed=17)
    assert rhos.shape == (39,)
    assert values.shape == (39, 1)
    assert values[0, 0] >= 0
