from tutorial import tut_1_1


def test_tut_1_1_mm1_arrival_service_processes_interact():
    trial = tut_1_1.run(stop_time=25.0, seed=11)

    assert trial.arrivals > 0
    assert trial.services > 0
    assert trial.avg_queue_length >= 0.0
