from tutorial import tut_4_0


def test_tut_4_0_empty_simulation_template_runs_control_events():
    result = tut_4_0.run_template(duration=10.0, seed=40)

    assert result["now"] == 10.0
    assert result["event_count"] == 0
