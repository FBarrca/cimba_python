from tutorial import tut_1_5


def test_tut_1_5_parameterized_trial_stores_results_on_trial_object():
    trial = tut_1_5.MM1Trial(arr_rate=0.6, srv_rate=1.0, warmup_time=20.0, duration=1500.0, seed=15)

    result = tut_1_5.run(trial)

    assert result is trial
    assert result.avg_queue_length > 0.0
    assert result.arrivals >= result.services
