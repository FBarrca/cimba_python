from tutorial import tut_4_2


def test_tut_4_2_more_large_berths_improve_serial_harbor_scenario():
    base = dict(
        mean_wind=5.0,
        reference_depth=15.0,
        arrival_rate=0.5,
        percent_large=1.0,
        num_tugs=6.0,
        num_berths_small=1.0,
        unload_avg_small=2.0,
        unload_avg_large=4.0,
        replications=1,
        warmup=0.0,
        duration=72.0,
        seed=42,
    )
    one_large = tut_4_2.harbor.experiment(**base, num_berths_large=1.0)
    two_large = tut_4_2.harbor.experiment(**base, num_berths_large=2.0)

    assert one_large.run() == 0
    assert two_large.run() == 0
    assert two_large["avg_time_large"][0] < one_large["avg_time_large"][0]
