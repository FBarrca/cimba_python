import cimba as cb
from tutorial.tut_2_1 import CheeseGame


def test_cat_interrupts_rodents_competing_for_cheese():
    game = CheeseGame()
    results = cb.Experiment(game, window=cb.Window(duration=100), seed=22).run()
    assert not results.failed.any()
    assert results[game].mice_grabbed[0, 0] > 0
    assert results[game].cat_chases[0, 0] > 0
    assert (results[game].mice_interrupted[0, 0] +
            results[game].rats_interrupted[0, 0]) > 0
    assert results[game].accounting_errors[0, 0] == 0
