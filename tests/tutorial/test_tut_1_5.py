from tutorial import tut_1_5


def test_tut_1_5_parameterized_trial_stores_results_on_trial_object():
    avg_queue_length = tut_1_5.run_mm1_trial(
        utilization=0.6,
        warmup=20.0,
        duration=1500.0,
        seed=15,
    )

    assert avg_queue_length > 0.0
