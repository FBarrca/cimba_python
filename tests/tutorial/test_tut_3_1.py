from tutorial import tut_3_1


def test_tut_3_1_visitor_jockeys_to_shorter_queue_and_gets_served():
    visitor = tut_3_1.run_jockeying_demo()

    assert visitor.status == "served"
    assert visitor.num_attractions_visited == 1
    assert visitor.waiting_time == 0.0
    assert visitor.riding_time == 2.0


def test_tut_3_1_visitor_reneges_when_timer_expires_before_service():
    visitor, queue_length = tut_3_1.run_reneging_demo()

    assert visitor.status == "reneged"
    assert queue_length == 0
