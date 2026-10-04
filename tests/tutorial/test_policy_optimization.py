import numpy as np

import cimba as cb
from tutorial.policy_optimization import CallCentre, main, tune_policies, tune_staff


def test_tuned_policies_are_compared_with_common_inputs():
    store, results, optima = tune_policies(replications=8, validation=12, evaluations=60, workers=2)
    assert not results.failed.any()
    assert len(optima) == 2
    assert results[store].fill_rate.shape == (2, 12)
    for best in optima:
        assert best.estimate.n == 12
        assert best.batches[-2].phase == 'selection'
        assert best.batches[-1].phase == 'estimation'
    np.testing.assert_array_equal(results[store].demand.rows(0, 0)[:10],
                                  results[store].demand.rows(1, 0)[:10])


def test_integer_staffing_matches_exhaustive_search():
    centre, best = tune_staff(replications=40, validation=80, workers=4)
    assert best[centre.staff] == 7
    exhaustive = CallCentre()
    exhaustive.staff = cb.sweep(*range(1, 21))
    results = cb.Experiment(exhaustive, replications=40,
                            window=cb.Window(warmup=50, duration=500), seed=5).run(workers=4)
    costs = cb.analysis.summary(results[exhaustive].cost).mean
    assert np.argmin(costs) + 1 == best[centre.staff]


def test_main_prints_the_documented_report(capsys):
    main()
    out = capsys.readouterr().out
    assert "MinMax: minimum=60, maximum=118" in out
    assert "chosen" in out
    assert "Call centre: 7 agents" in out
