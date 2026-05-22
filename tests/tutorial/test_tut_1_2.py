from tutorial import tut_1_2


def test_tut_1_2_stop_event_ends_infinite_processes():
    result = tut_1_2.run(stop_time=3.5, seed=12)

    assert result["ticks"] == [1.0, 2.0, 3.0]
    assert result["event_count"] == 0
    assert result["now"] == 3.5
