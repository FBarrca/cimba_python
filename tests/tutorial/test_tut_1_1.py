from tutorial import tut_1_1


def test_tut_1_1_mm1_arrival_service_processes_interact():
    exp = tut_1_1.model.experiment(
        utilization=[0.75],
        replications=1,
        duration=25.0,
        warmup=0.0,
        seed=11,
    )

    assert exp.run() == 0
    assert exp["avg_queue_length"][0] >= 0.0
