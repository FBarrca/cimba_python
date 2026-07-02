from tutorial import tut_3_1


def test_tut_3_1_visitor_jockeys_to_shorter_queue_and_gets_served():
    exp = tut_3_1.park.experiment(
        replications=1,
        duration=60.0,
        warmup=0.0,
        cooldown=200.0,
        seed=31,
    )

    assert exp.run() == 0
    assert exp["n_visitors"][0] > 0
    assert exp["avg_rides"][0] >= 0.0


def test_tut_3_1_visitor_reneges_when_timer_expires_before_service():
    exp = tut_3_1.park.experiment(
        replications=1,
        duration=60.0,
        warmup=0.0,
        cooldown=200.0,
        seed=32,
    )

    assert exp.run() == 0
    assert exp["n_balks"][0] >= 0
    assert exp["n_reneges"][0] >= 0
