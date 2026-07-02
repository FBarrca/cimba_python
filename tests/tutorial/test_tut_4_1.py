from tutorial import tut_4_1


def test_tut_4_1_harbor_ship_waits_for_environment_and_resources():
    exp = tut_4_1.harbor.experiment(
        mean_wind=5.0,
        reference_depth=15.0,
        arrival_rate=1.0,
        percent_large=0.0,
        num_tugs=4.0,
        num_berths_small=2.0,
        num_berths_large=1.0,
        unload_avg_small=2.0,
        unload_avg_large=3.0,
        replications=1,
        warmup=0.0,
        duration=48.0,
        seed=41,
    )

    assert exp.run() == 0
    assert exp["n_small"][0] > 0
    assert exp["avg_time_small"][0] > 0.0
    assert exp["tug_util"][0] > 0.0
