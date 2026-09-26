"""Compare inventory policies with polymorphic functions and a reference sweep.

A store reviews its stock every day and asks its replenishment policy how much
to order. The policy is a model whose ``order_quantity`` is a ``@cb.function``;
each policy class overrides it. The store's process never changes: sweeping
the ``policy`` reference runs every candidate in one experiment, on the same
demand and lead times (common random numbers), so the comparison is paired.
"""

import cimba as cb
from cimba import inputs


class Policy(cb.Model):
    """How much to order, given stock on hand and the inventory position."""

    @cb.function
    def order_quantity(self, on_hand: float, position: float) -> float:
        return 0.0


class OrderUpTo(Policy):
    """Every day, order back up to a fixed level."""

    level: cb.Param[float] = 90.0

    @cb.function
    def order_quantity(self, on_hand: float, position: float) -> float:
        return max(0.0, self.level - position)


class MinMax(Policy):
    """When the position falls to the minimum, order up to the maximum."""

    minimum: cb.Param[float] = 40.0
    maximum: cb.Param[float] = 120.0

    @cb.function
    def order_quantity(self, on_hand: float, position: float) -> float:
        if position <= self.minimum:
            return self.maximum - position
        return 0.0


class FixedQuantity(Policy):
    """When the position falls to the reorder point, order a fixed batch."""

    reorder_point: cb.Param[float] = 45.0
    batch: cb.Param[float] = 80.0

    @cb.function
    def order_quantity(self, on_hand: float, position: float) -> float:
        return self.batch if position <= self.reorder_point else 0.0


class Delivery(cb.Model):
    """An order on its way: it arrives after a random lead time."""

    store: cb.Ref["Store"]
    quantity: cb.State[float]

    @cb.process
    def arrive(self):
        cb.hold(self.store.lead_time.next())
        self.store.on_hand += self.quantity
        cb.release(self)


class Store(cb.Model):
    policy: cb.Ref[Policy]
    demand: cb.Input[float] = inputs.dist.gamma(shape=2.0, scale=6.0)
    lead_time: cb.Input[float] = inputs.dist.uniform(low=2.0, high=5.0)
    on_hand: cb.State[float] = 80.0
    position: cb.State[float] = 80.0
    daily_demand: cb.Dataset
    daily_sales: cb.Dataset
    daily_stock: cb.Dataset
    daily_orders: cb.Dataset
    fill_rate: cb.Output[float]
    mean_stock: cb.Output[float]
    orders_per_week: cb.Output[float]

    @cb.process
    def trade(self):
        while True:
            cb.hold(1.0)
            demand = self.demand.next()
            sold = min(demand, self.on_hand)            # unmet demand is lost
            self.on_hand -= sold
            self.position -= sold                     # position = on hand + on order
            self.daily_demand.record(demand)
            self.daily_sales.record(sold)
            self.daily_stock.record(self.on_hand)
            quantity = self.policy.order_quantity(self.on_hand, self.position)
            if quantity > 0.0:
                cb.spawn(Delivery, store=self, quantity=quantity)
                self.position += quantity
                self.daily_orders.record(1.0)
            else:
                self.daily_orders.record(0.0)

    @cb.on_end
    def measure(self):
        self.fill_rate = self.daily_sales.sample_mean() / self.daily_demand.sample_mean()
        self.mean_stock = self.daily_stock.sample_mean()
        self.orders_per_week = 7.0 * self.daily_orders.sample_mean()


class Study(cb.Model):
    """Owns every candidate policy and the store that uses one of them."""

    candidates: list[Policy]
    store: Store

    def __init__(self):
        self.candidates = [OrderUpTo(), MinMax(), FixedQuantity()]
        self.store = Store()


def swept_study() -> Study:
    """A study whose store tries every candidate policy, one per design point."""
    study = Study()
    study.store.policy = cb.sweep(*study.candidates)  # pyright: ignore[reportAttributeAccessIssue]
    return study


def compare_policies(*, replications: int = 50, seed: int = 11):
    study = swept_study()
    choice = study.store.policy
    results = cb.Experiment(
        study, replications=replications,
        window=cb.Window(warmup=30.0, duration=365.0), seed=seed,
    ).run()
    if results.failed.any():
        raise RuntimeError(f"{results.failed.sum()} policy trials failed")
    return study, choice, results


def main() -> None:
    study, choice, results = compare_policies()
    store = results[study.store]
    fill = cb.analysis.summary(store.fill_rate)
    stock = cb.analysis.summary(store.mean_stock)
    orders = cb.analysis.summary(store.orders_per_week)
    print(f"{'policy':<14}{'fill rate':>11}{'mean stock':>12}{'orders/wk':>11}")
    for index, policy in enumerate(results.levels(choice)):
        print(f"{type(policy).__name__:<14}{fill.mean[index]:>11.3f}"
              f"{stock.mean[index]:>12.1f}{orders.mean[index]:>11.2f}")
    for index in (1, 2):
        c = cb.analysis.compare(store.fill_rate, a=0, b=index)
        print(f"fill rate, {type(results.levels(choice)[index]).__name__} - OrderUpTo: "
              f"{c.difference:+.4f} (95% CI {c.lower:+.4f} .. {c.upper:+.4f})")


if __name__ == "__main__":
    main()
