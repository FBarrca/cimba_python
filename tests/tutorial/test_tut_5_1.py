from tutorial import tut_5_1


def test_tut_5_1_reduced_awacs_target_sensor_loop_collects_detections():
    result = tut_5_1.run_awacs_demo(seed=51, duration=5.0, num_targets=3)

    assert result["count"] > 0
    assert 0.0 <= result["mean"] <= 1.0
