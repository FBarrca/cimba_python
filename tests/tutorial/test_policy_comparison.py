import cimba as cb
from tutorial.policy_comparison import (
    FixedQuantity, MinMax, OrderUpTo, compare_policies,
)


def test_policies_are_compared_in_one_experiment():
    study, choice, results = compare_policies(replications=6)
    assert [type(p) for p in results.levels(choice)] == [OrderUpTo, MinMax, FixedQuantity]
    store = results[study.store]
    assert store.fill_rate.shape == (3, 6)
    orders = cb.analysis.summary(store.orders_per_week).mean
    assert orders[0] > 5 * orders[1]            # daily review orders far more often
    comparison = cb.analysis.compare(store.fill_rate, a=0, b=1)
    assert comparison.n == 6
