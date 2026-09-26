from tutorial.tut_1_5 import run_mm1_trial


def test_composed_station_runs():
    assert run_mm1_trial(utilization=0.6, warmup=20,
                         duration=1_500, seed=15) > 0
