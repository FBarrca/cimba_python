from tutorial import tut_2_1


def test_tut_2_1_preemption_wakes_lower_priority_holder():
    exp = tut_2_1.game.experiment(
        replications=1,
        duration=100.0,
        warmup=0.0,
        seed=21,
    )

    assert exp.run() == 0
    assert exp["accounting_errors"][0] == 0
    assert exp["mice_grabbed"][0] > 0


def test_tut_2_1_cat_chases_interrupt_rodents():
    exp = tut_2_1.game.experiment(
        replications=1,
        duration=100.0,
        warmup=0.0,
        seed=22,
    )

    assert exp.run() == 0
    assert exp["cat_chases"][0] > 0
    assert exp["mice_interrupted"][0] + exp["rats_interrupted"][0] > 0
