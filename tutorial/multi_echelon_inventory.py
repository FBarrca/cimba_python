"""Multi-echelon inventory with source-agnostic daily demand.

The six-node network keeps a source facility and five stocking facilities.
One joint stationary bootstrap drives their related demand histories; an
independent bootstrap supplies shipment delays. Sources can be replaced with
traces or fitted models without changing the compiled facility processes.
"""

from pathlib import Path

import numpy as np

import cimba as cb
from cimba import inputs


SOURCE_NODE = 0
NUM_NODES = 6
STOCKING_NODES = 5
EPSILON = 1.0e-5
REORDER_POINT_TOLERANCE = 0.05
DURATION = 360.0
WARMUP = 0.0

BASE_LEAD_TIME = np.array([0.0, 3.0, 4.0, 4.0, 2.0, 2.0])
BASE_STOCK = np.array([10000.0, 3000.0, 600.0, 900.0, 300.0, 600.0])
REORDER_POINT = np.array([0.0, 1000.0, 250.0, 200.0, 150.0, 200.0])
INITIAL_INVENTORY = 0.9 * BASE_STOCK


class Order(cb.Model):
    requester: cb.Ref["Facility"]
    quantity: cb.State[float]


class Facility(cb.Model):
    network: cb.Ref["MultiEchelonInventory"]
    node: cb.Param[int]
    upstream: cb.Ref["Facility"] | None
    base_stock: cb.Param[float]
    reorder_point: cb.Param[float]
    initial_inventory: cb.Param[float]
    base_lead_time: cb.Param[float]
    on_hand: cb.State[float] = 0.0
    inventory_position: cb.State[float] = 0.0
    backorder: cb.State[float] = 0.0
    total_demand: cb.State[float] = 0.0
    total_shipped: cb.State[float] = 0.0
    total_late_sales: cb.State[float] = 0.0
    on_hand_total: cb.State[float] = 0.0
    on_hand_samples: cb.State[int] = 0
    avg_on_hand: cb.Output[float]
    service_level: cb.Output[float]
    orders: cb.Store[Order]

    def __init__(self, network, node, upstream, *, base_stock,
                 reorder_point, initial_inventory, base_lead_time):
        self.network = network
        self.node = node
        self.upstream = upstream
        self.base_stock = float(base_stock)
        self.reorder_point = float(reorder_point)
        self.initial_inventory = float(initial_inventory)
        self.base_lead_time = float(base_lead_time)


class SourceFacility(Facility):
    """External source with unlimited stock."""

    @cb.process
    def fulfill_orders(self):
        while True:
            order = self.orders.get()
            cb.spawn(Shipment, network=self.network,
                     requester=order.requester, quantity=order.quantity)
            cb.release(order)

    @cb.on_end
    def facility_stats(self):
        self.avg_on_hand = 0.0
        self.service_level = 1.0


class StockingFacility(Facility):
    """Inventory-holding node with a daily series and replenishment."""

    demand: cb.Series[float] = cb.Series(step=1.0, origin=1.0)

    @cb.on_start
    def initialize(self):
        self.on_hand = self.initial_inventory
        self.inventory_position = self.initial_inventory

    @cb.process
    def place_order(self):
        while True:
            cb.hold(1.0)
            threshold = self.reorder_point * (1.0 + REORDER_POINT_TOLERANCE)
            if self.inventory_position <= threshold:
                quantity = self.base_stock - self.on_hand
                if quantity > 0.0:
                    order = cb.spawn(Order, requester=self, quantity=quantity)
                    if self.upstream is not None:
                        self.upstream.orders.put(order)
                    self.inventory_position += quantity

    @cb.process
    def fulfill_orders(self):
        while True:
            order = self.orders.get()
            quantity = order.quantity
            while self.on_hand < quantity:
                cb.hold(1.0)
            self.on_hand -= quantity
            self.inventory_position -= quantity
            cb.spawn(Shipment, network=self.network,
                     requester=order.requester, quantity=quantity)
            cb.release(order)

    @cb.process
    def serve_customer(self):
        while True:
            self.on_hand_total += self.on_hand
            self.on_hand_samples += 1
            cb.hold(1.0)
            demand = self.demand.now()
            self.total_demand += demand
            if self.network.backorder >= 0.5:
                shipped = min(demand + self.backorder, self.on_hand)
                self.on_hand -= shipped
                self.inventory_position -= shipped
                remaining = demand - shipped
                self.backorder += remaining
                if remaining > 0.0:
                    self.total_late_sales += remaining
            else:
                shipped = min(demand, self.on_hand)
                self.total_shipped += shipped
                self.on_hand -= shipped
                self.inventory_position -= shipped

    @cb.on_end
    def facility_stats(self):
        if self.on_hand_samples > 0:
            self.avg_on_hand = self.on_hand_total / self.on_hand_samples
        else:
            self.avg_on_hand = self.on_hand
        demand = self.total_demand + EPSILON
        if self.network.backorder >= 0.5:
            self.service_level = 1.0 - self.total_late_sales / demand
        else:
            self.service_level = self.total_shipped / demand


class Shipment(cb.Model):
    network: cb.Ref["MultiEchelonInventory"]
    requester: cb.Ref[Facility]
    quantity: cb.State[float]

    @cb.process
    def deliver(self):
        delay = self.network.lead_time_delay.next()
        lead_time = self.requester.base_lead_time + delay
        if lead_time > 0.0:
            cb.hold(lead_time)
        self.requester.on_hand += self.quantity
        self.network.completed_shipments.put(self)


class MultiEchelonInventory(cb.Model):
    backorder: cb.Param[float] = 0.0
    lead_time_delay: cb.Input[float] = inputs.trace([0.0], on_exhausted="wrap")
    completed_shipments: cb.Store[Shipment]
    facilities: list[Facility]

    def __init__(self, *, backorder=0.0, base_stock=BASE_STOCK,
                 reorder_point=REORDER_POINT,
                 initial_inventory=INITIAL_INVENTORY,
                 base_lead_time=BASE_LEAD_TIME):
        self.backorder = float(backorder)
        source = SourceFacility(
            self, SOURCE_NODE, None,
            base_stock=base_stock[0], reorder_point=reorder_point[0],
            initial_inventory=initial_inventory[0],
            base_lead_time=base_lead_time[0],
        )
        dc = StockingFacility(
            self, 1, source,
            base_stock=base_stock[1], reorder_point=reorder_point[1],
            initial_inventory=initial_inventory[1],
            base_lead_time=base_lead_time[1],
        )
        second = StockingFacility(
            self, 3, dc,
            base_stock=base_stock[3], reorder_point=reorder_point[3],
            initial_inventory=initial_inventory[3],
            base_lead_time=base_lead_time[3],
        )
        nodes = [source, dc, None, second, None, None]
        for node, upstream in ((2, dc), (4, second), (5, second)):
            nodes[node] = StockingFacility(
                self, node, upstream,
                base_stock=base_stock[node], reorder_point=reorder_point[node],
                initial_inventory=initial_inventory[node],
                base_lead_time=base_lead_time[node],
            )
        self.facilities = nodes

    @cb.process
    def reclaim_shipments(self):
        while True:
            cb.release(self.completed_shipments.get())


def load_data(data_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    root = (Path(__file__).with_name("data") / "multi_echelon_inventory"
            if data_dir is None else Path(data_dir))
    return (np.loadtxt(root / "demandData.csv", delimiter=",", skiprows=1),
            np.loadtxt(root / "leadTimeExtraDays.csv", delimiter=","))


def main() -> int:
    demand, lead_time_delay = load_data()
    network = MultiEchelonInventory()
    mean_block = round(demand.shape[0] ** (1.0 / 3.0))
    joint = inputs.bootstrap.joint(
        {node: demand[:, node - 1] for node in range(1, NUM_NODES)},
        mean_block=mean_block, tag="facility-demand",
    )
    for node in range(1, NUM_NODES):
        network.facilities[node].demand = joint[node]
    network.lead_time_delay = inputs.bootstrap.iid(lead_time_delay)
    result = cb.Experiment(
        network, replications=20,
        window=cb.Window(warmup=WARMUP, duration=DURATION), seed=123,
    ).run()
    if result.failed.any():
        raise RuntimeError(f"{result.failed.sum()} inventory trials failed")
    print("Average on-hand by node:", np.round([
        result[facility].avg_on_hand.values.mean()
        for facility in network.facilities], 3))
    print("Service level by node:", np.round([
        result[facility].service_level.values.mean()
        for facility in network.facilities], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
