import cimba
from tutorial import tut_2_1


def test_tut_2_1_preemption_wakes_lower_priority_holder():
    assert tut_2_1.run_preemption_demo() == [
        ("mouse-acquired", 0.0, 1),
        ("rat-preempted", 1.0, 1, 0),
        ("mouse-hold-returned", 1.0, cimba.PREEMPTED, 0),
    ]


def test_tut_2_1_waiting_process_can_be_interrupted_by_another_process():
    assert tut_2_1.run_interruption_demo() == [("waiting-mouse", 0.5, 77, 0, 4)]
