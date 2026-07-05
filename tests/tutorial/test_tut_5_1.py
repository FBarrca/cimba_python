from tutorial import tut_5_1


def test_tut_5_1_runs_short_assembly_line_experiment(tmp_path):
    model = tut_5_1.build_model(tmp_path)
    exp = model.experiment(
        replications=1,
        duration=500.0,
        warmup=0.0,
        seed=tut_5_1.RANDOM_SEED,
    )

    assert exp.run() == 0
    assert exp["total_parts_produced"][0] > 0
    assert exp["avg_cycle_time"][0] > 0.0
    assert exp["station_2__utilization"][0] > exp["station_3__utilization"][0]
