from tutorial import tut_1_2


def test_tut_1_2_stop_event_ends_infinite_processes():
    exp = tut_1_2.model.experiment(
        utilization=[0.75],
        replications=1,
        duration=3.5,
        warmup=0.0,
        seed=12,
    )

    assert exp.run() == 0
    assert exp["avg_queue_length"][0] >= 0.0
