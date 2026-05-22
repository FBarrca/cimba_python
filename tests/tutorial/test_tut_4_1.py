from tutorial import tut_4_1


def test_tut_4_1_harbor_ship_waits_for_environment_and_resources():
    result = tut_4_1.run_harbor_trial(seed=41)

    assert len(result["small_system_times"]) == 1
    assert result["small_system_times"][0] > 3.0
    assert result["tugs_available"] == 2
    assert result["small_berths_available"] == 1
