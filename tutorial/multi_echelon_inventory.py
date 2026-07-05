"""Multi-echelon inventory policy in Cimba Python.

This module implements a simulation-side base policy. It does not run a
simulation-optimization loop.

The network has six nodes:

* node 0 is an external source with infinite inventory,
* nodes 1-5 are stocking facilities,
* 0 -> 1, 1 -> 2, 1 -> 3, 3 -> 4, and 3 -> 5 are replenishment arcs.

Historical demand and lead-time delay observations are bootstrap-resampled
outside the simulation with ``cimba.bootstrap`` and replayed as Cimba trace
fields: facility demands with a joint stationary bootstrap (one set of block
draws for all facilities, preserving autocorrelation and cross-facility
correlation), lead-time delays with the ordinary i.i.d. bootstrap (the
observations are independent). See docs/advanced/bootstrapping.rst for the
method survey.

Usage with the bundled data:

    python tutorial/multi_echelon_inventory.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import cimba.sim as sim
from cimba import bootstrap


SOURCE_NODE = 0
NUM_NODES = 6
STOCKING_NODES = 5
EPSILON = 1.0e-5
REORDER_POINT_TOLERANCE = 0.05
DURATION = 360.0
WARMUP = 0.0

# Parameters for each stocking facility (0-6)
BASE_LEAD_TIME = np.array([0.0, 3.0, 4.0, 4.0, 2.0, 2.0])
BASE_STOCK = np.array([10000.0, 3000.0, 600.0, 900.0, 300.0, 600.0])
REORDER_POINT = np.array([0.0, 1000.0, 250.0, 200.0, 150.0, 200.0])
INITIAL_INVENTORY = 0.9 * BASE_STOCK


class Shipment(sim.Struct):
    requester: int
    quantity: float


class Facility(sim.Component):
    demand: sim.Trace

    on_hand: sim.FloatState
    inventory_position: sim.FloatState
    backorder: sim.FloatState
    total_demand: sim.FloatState
    total_shipped: sim.FloatState
    total_late_sales: sim.FloatState
    on_hand_total: sim.FloatState
    on_hand_samples: sim.State

    avg_on_hand: sim.Output
    service_level: sim.Output

    order_requesters: sim.Store
    order_quantities: sim.Store

    def __init__(self, node: int, upstream: int, is_source: int):
        self.node = node
        self.upstream = upstream
        self.is_source = is_source

    @sim.process(priority=10)
    def initialize(self, env):
        initial_inventory = sim.Trace(env.initial_inventory)
        self.on_hand = initial_inventory[self.node]
        self.inventory_position = initial_inventory[self.node]

    @sim.process
    def place_order(self, env):
        base_stock = sim.Trace(env.base_stock)
        reorder_point = sim.Trace(env.reorder_point)
        while True:
            sim.hold(1.0)
            if self.is_source == 1:
                continue  # Source node does not place orders

            # Inventory position includes stock already ordered but not arrived.
            threshold = reorder_point[self.node] \
                * (1.0 + REORDER_POINT_TOLERANCE)
            if self.inventory_position <= threshold:
                # Order enough to bring this facility back up to base stock.
                quantity = base_stock[self.node] - self.on_hand
                if quantity > 0.0:
                    upstream = self.upstream
                    sim.store_put(
                        env.facilities[upstream].order_requesters,
                        self.node,
                    )
                    sim.store_put(
                        env.facilities[upstream].order_quantities,
                        sim.f2i(quantity),
                    )
                    # Count the order immediately so we do not reorder it again.
                    self.inventory_position += quantity

    @sim.process
    def fulfill_orders(self, env):
        while True:
            if sim.store_length(self.order_requesters) == 0:
                sim.hold(1.0)
            else:
                # These are replenishment orders from downstream facilities.
                requester = sim.store_take(self.order_requesters)
                quantity = sim.i2f(sim.store_take(self.order_quantities))

                if self.is_source == 1:
                    # External supply is unlimited, so nothing is deducted.
                    handle = sim.spawn(env.shipment, env)
                    shipment = Shipment(handle)
                    shipment.requester = requester
                    shipment.quantity = quantity
                    continue

                # Stocking nodes may have to wait until enough stock arrives.
                shipped_now = min(quantity, self.on_hand)
                self.on_hand -= shipped_now
                self.inventory_position -= shipped_now

                remaining = quantity - shipped_now
                if remaining > EPSILON:
                    while self.on_hand < remaining:
                        sim.hold(1.0)
                    self.on_hand -= remaining
                    self.inventory_position -= remaining

                handle = sim.spawn(env.shipment, env)
                shipment = Shipment(handle)
                shipment.requester = requester
                shipment.quantity = quantity

    @sim.process
    def serve_customer(self, env):
        # The trace holds this trial's bootstrap trajectory; replay it in
        # order so the temporal structure of the resample survives.
        trajectory = sim.Trace(self.demand)
        day = 0
        while True:
            if self.is_source == 1:
                sim.suspend()
            else:
                # Customer demand applies only to the stocking nodes.
                self.on_hand_total += self.on_hand
                self.on_hand_samples += 1
                sim.hold(1.0)

                demand = trajectory[day]
                day += 1
                self.total_demand += demand

                if env.backorder >= 0.5:
                    # Backorder mode counts demand not filled immediately.
                    shipment = min(demand + self.backorder, self.on_hand)
                    self.on_hand -= shipment
                    self.inventory_position -= shipment

                    backorder_delta = demand - shipment
                    self.backorder += backorder_delta
                    if backorder_delta > 0.0:
                        self.total_late_sales += backorder_delta
                else:
                    # Lost-sales mode counts only units shipped on demand.
                    shipment = min(demand, self.on_hand)
                    self.total_shipped += shipment
                    self.on_hand -= shipment
                    self.inventory_position -= shipment

    @sim.collect
    def facility_stats(self, env):
        if self.is_source == 1:
            self.avg_on_hand = 0.0
            self.service_level = 1.0
        else:
            if self.on_hand_samples > 0:
                self.avg_on_hand = self.on_hand_total / self.on_hand_samples
            else:
                self.avg_on_hand = self.on_hand

            demand = self.total_demand + EPSILON
            if env.backorder >= 0.5:
                self.service_level = 1.0 - self.total_late_sales / demand
            else:
                self.service_level = self.total_shipped / demand


class MultiEchelonInventory(sim.Model):
    backorder: sim.Param

    base_stock: sim.Trace
    reorder_point: sim.Trace
    initial_inventory: sim.Trace
    base_lead_time: sim.Trace
    lead_time_delay: sim.Trace

    shipment: sim.Spawnable
    completed_shipments: sim.Store
    lead_time_cursor: sim.State

    facilities: list[Facility] = [
        Facility(0, -1, 1),
        Facility(1, 0, 0),
        Facility(2, 1, 0),
        Facility(3, 1, 0),
        Facility(4, 3, 0),
        Facility(5, 3, 0),
    ]


model = MultiEchelonInventory("multi-echelon-inventory")


@model.process
def shipment(env: MultiEchelonInventory, shipment: Shipment):
    lead_time_delay = sim.Trace(env.lead_time_delay)
    base_lead_time = sim.Trace(env.base_lead_time)
    requester = shipment.requester
    # Each shipment consumes the next resampled delay from the trace.
    draw = env.lead_time_cursor
    env.lead_time_cursor += 1
    delay = lead_time_delay[draw]
    lead_time = base_lead_time[requester] + delay
    if lead_time > 0.0:
        sim.hold(lead_time)
    # The replenishment arrives after its sampled lead time.
    env.facilities[requester].on_hand += shipment.quantity
    sim.store_put(env.completed_shipments, sim.current())


@model.process
def reclaim_shipments(env: MultiEchelonInventory):
    while True:
        handle = sim.store_take(env.completed_shipments)
        sim.despawn(handle)


def load_data(
    data_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load demand and lead-time observations."""

    root = (
        Path(__file__).with_name("data") / "multi_echelon_inventory"
        if data_dir is None
        else Path(data_dir)
    )
    return (
        np.loadtxt(root / "demandData.csv", delimiter=",", skiprows=1),
        np.loadtxt(root / "leadTimeExtraDays.csv", delimiter=","),
    )


def main() -> int:
    demand, lead_time_delay = load_data()

    # One demand value per simulated day; size the resamples to cover the
    # whole recording window (warmup + duration).
    horizon = int(WARMUP + DURATION) + 1

    # The facility demand histories are related series, so resample them
    # jointly: one set of stationary-bootstrap block draws drives every
    # facility, preserving autocorrelation within each series and the
    # cross-facility correlation that stresses shared upstream capacity.
    # Mean block length follows the n**(1/3) rule of thumb.
    mean_block = round(demand.shape[0] ** (1.0 / 3.0))
    demand_gens = bootstrap.joint(
        {f"facility_{node}": demand[:, node - 1]
         for node in range(1, NUM_NODES)},
        length=horizon,
        name="demand",
        mean_block=mean_block,
    )
    # Node 0 has no demand, so we prepend a zero array.
    facility_demand = [np.zeros(horizon)] + [
        demand_gens[f"facility_{node}"] for node in range(1, NUM_NODES)
    ]

    # Lead-time extra days are independent observations, so the ordinary
    # bootstrap applies. Each shipment consumes one draw; the stocking
    # facilities order at most once per day, which bounds the draws needed.
    lead_time_gen = bootstrap.iid(
        lead_time_delay, length=STOCKING_NODES * horizon)

    exp = model.experiment(
        backorder=0.0,
        base_stock=BASE_STOCK,
        reorder_point=REORDER_POINT,
        initial_inventory=INITIAL_INVENTORY,
        base_lead_time=BASE_LEAD_TIME,
        lead_time_delay=lead_time_gen,
        facilities__demand=facility_demand,
        replications=20,
        duration=DURATION,
        warmup=WARMUP,
        seed=123,
    )
    failures = exp.run()
    if failures:
        raise RuntimeError(f"{failures} trial(s) failed")

    print(
        "Average on-hand by node:",
        np.round(exp["facilities__avg_on_hand"].mean(axis=0), 3),
    )
    print(
        "Service level by node:",
        np.round(exp["facilities__service_level"].mean(axis=0), 4),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
