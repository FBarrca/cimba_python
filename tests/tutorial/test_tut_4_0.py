from tutorial import tut_4_0


def test_tut_4_0_empty_simulation_template_runs_control_events():
    exp = tut_4_0.model.experiment(
        replications=1,
        duration=10.0,
        warmup=0.0,
        seed=40,
    )

    assert exp.run() == 0
    assert exp["result"][0] == 0.0
