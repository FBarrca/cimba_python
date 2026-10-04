"""Chapter 7: find good policy parameters and staffing levels by optimization.

Two inventory policies are tuned for the same store, then compared on fresh
trials. A call centre's staffing level shows an integer decision.
"""
# Decisions replace Param values on the host; pyright types those fields as floats.
# pyright: reportAttributeAccessIssue=false

import cimba as cb
from cimba import inputs

from tutorial.policy_comparison import MinMax, OrderUpTo, Store

WINDOW = cb.Window(warmup=30.0, duration=365.0)


def store_cost(results, store):
    """Cost of one trial: average stock, plus ordering, plus lost sales."""
    outputs = results[store]
    return (outputs.mean_stock
            + 20.0 * outputs.orders_per_week
            + 400.0 * (1.0 - outputs.fill_rate))


def tunable_min_max():
    policy = MinMax()
    policy.minimum = cb.decision(0.0, 100.0, step=1.0)
    policy.maximum = cb.decision(40.0, 250.0, step=1.0)
    return policy


def tunable_order_up_to():
    policy = OrderUpTo()
    policy.level = cb.decision(40.0, 250.0, step=1.0)
    return policy


def tune(policy, *, replications=32, evaluations=300, validation=None, workers=None):
    store = Store()
    store.policy = policy
    study = cb.Optimization(
        store,
        minimize=lambda results: store_cost(results, store),
        replications=replications,
        window=WINDOW,
        seed=11,
    )
    return study.run(evaluations=evaluations, validation=validation, workers=workers)


def compare(policies, *, replications=128, workers=None):
    store = Store()
    store.policy = cb.sweep(*policies)
    results = cb.Experiment(store, replications=replications, window=WINDOW,
                            seed=12).run(workers=workers)
    return store, results


def tune_policies(*, replications=32, evaluations=300, validation=None, workers=None):
    """Tune both policies, apply the answers, and compare them on fresh trials."""
    policies = (tunable_min_max(), tunable_order_up_to())
    optima = []
    for policy in policies:
        best = tune(policy, replications=replications, evaluations=evaluations,
                    validation=validation, workers=workers)
        best.apply()
        optima.append(best)
    store, results = compare(policies, replications=validation or 128, workers=workers)
    return store, results, tuple(optima)


class Caller(cb.Model):
    centre: cb.Ref["CallCentre"]

    @cb.process
    def call(self):
        arrived = cb.now()
        self.centre.agents.get(1)               # wait for a free agent
        self.centre.waits.record(cb.now() - arrived)
        cb.hold(self.centre.handling.next())
        self.centre.agents.put(1)               # the agent is free again
        cb.release(self)


class CallCentre(cb.Model):
    staff: cb.Param[int] = 3
    gaps: cb.Input[float] = inputs.dist.exponential(mean=0.25)
    handling: cb.Input[float] = inputs.dist.exponential(mean=1.0)
    agents: cb.Container                        # one token per free agent
    waits: cb.Dataset
    cost: cb.Output[float]

    @cb.on_start
    def open(self):
        self.agents.put(self.staff)

    @cb.process
    def calls(self):
        while True:
            cb.hold(self.gaps.next())
            cb.spawn(Caller, centre=self)

    @cb.on_end
    def measure(self):
        self.cost = 20.0 * self.staff + 400.0 * self.waits.sample_mean()


def tune_staff(*, replications=40, validation=None, workers=None):
    centre = CallCentre()
    centre.staff = cb.decision(1, 20)
    best = cb.Optimization(
        centre,
        minimize=lambda results: results[centre].cost,
        replications=replications,
        window=cb.Window(warmup=50.0, duration=500.0),
        seed=5,
    ).run(validation=validation, workers=workers)
    return centre, best


def describe(estimate):
    return f"{estimate.mean:.2f} (95% CI {estimate.lower:.2f} to {estimate.upper:.2f})"


def main():
    policy = tunable_min_max()
    best = tune(policy)
    print(f"MinMax: minimum={best[policy.minimum]:g}, maximum={best[policy.maximum]:g}, "
          f"cost {describe(best.estimate)}")
    print(f"  searched {best.meta.evaluations} candidates in {best.meta.generations} generations")
    print("  finalists (minimum, maximum, cost, difference from the choice):")
    for finalist in best.finalists:
        difference = finalist.difference
        compared = ("chosen" if finalist.values == best.values else
                    f"{difference.difference:+5.2f} ({difference.lower:+.2f} to {difference.upper:+.2f})")
        print(f"    {finalist.values[policy.minimum]:4g} {finalist.values[policy.maximum]:4g}"
              f"  {finalist.selection_mean:6.2f}  {compared}")

    store, results, optima = tune_policies()
    costs = store_cost(results, store)
    summary = cb.analysis.summary(costs)
    print("Tuned policies on fresh trials:")
    for index, tuned in enumerate(results.levels(store.policy)):
        print(f"  {type(tuned).__name__:<10} cost {summary.mean[index]:.2f}")
    difference = cb.analysis.compare(costs, a=0, b=1)
    print(f"  OrderUpTo - MinMax: {difference.difference:+.2f} "
          f"(95% CI {difference.lower:+.2f} to {difference.upper:+.2f})")

    centre, best = tune_staff()
    print(f"Call centre: {best[centre.staff]} agents, cost {describe(best.estimate)}")


if __name__ == "__main__":
    main()
