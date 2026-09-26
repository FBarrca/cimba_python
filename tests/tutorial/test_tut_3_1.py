import cimba as cb
from tutorial.tut_3_1 import Park


def test_park_visitors_ride_and_renege():
    park = Park(closing=60)
    results = cb.Experiment(
        park, window=cb.Window(duration=60, cooldown=200), seed=31,
    ).run()
    assert not results.failed.any()
    assert results[park].n_visitors[0, 0] > 0
    assert results[park].avg_rides[0, 0] >= 0
    assert results[park].n_reneges[0, 0] >= 0
