import cimba as cb
from tutorial.tut_4_1 import Harbor


def test_ships_wait_for_weather_and_harbor_resources():
    harbor = Harbor(mean_wind=5, reference_depth=15, arrival_rate=1,
                    percent_large=0, num_tugs=4, num_berths_small=2,
                    num_berths_large=1, unload_avg_small=2,
                    unload_avg_large=3)
    results = cb.Experiment(
        harbor, window=cb.Window(duration=48), seed=41,
    ).run()
    assert not results.failed.any()
    assert results[harbor].n_small[0, 0] > 0
    assert results[harbor].avg_time_small[0, 0] > 0
    assert results[harbor].tug_util[0, 0] > 0
