import numpy as np

import cimba as cb
from tutorial.policy_comparison import (
    FixedQuantity, MinMax, OrderUpTo, PeriodicReview, compare_policies,
)


def test_policies_are_compared_in_one_experiment():
    store, results = compare_policies(replications=6)
    policies = results.levels(store.policy)
    assert [type(p) for p in policies] == [OrderUpTo, MinMax, FixedQuantity, PeriodicReview]
    outcome = results[store]
    assert outcome.fill_rate.shape == (4, 6)
    orders = cb.analysis.summary(outcome.orders_per_week).mean
    assert orders[0] > 5 * orders[1]            # daily review orders far more often
    assert abs(orders[3] - 1.0) < 0.05          # the weekly clock runs in its own point
    # Every policy sees the same demand stream (common random numbers).
    demand = [outcome.demand.rows(point, 0)[:10] for point in range(4)]
    assert all(np.array_equal(demand[0], other) for other in demand[1:])
    assert cb.analysis.compare(outcome.fill_rate, a=0, b=1).n == 6
