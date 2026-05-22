from tutorial import tut_1_6


def test_tut_1_6_serial_experiment_sweep_preserves_utilization_trend():
    rows = tut_1_6.run_experiment(rhos=(0.25, 0.75), replications=1, duration=2500.0)

    assert rows[0]["avg_queue_length"] < rows[1]["avg_queue_length"]
