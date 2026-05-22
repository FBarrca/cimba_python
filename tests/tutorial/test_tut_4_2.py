from tutorial import tut_4_2


def test_tut_4_2_more_large_berths_improve_serial_harbor_scenario():
    result = tut_4_2.run_scenarios()

    assert result["two_large_berths"] < result["one_large_berth"]
