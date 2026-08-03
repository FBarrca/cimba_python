"""Pyright fixture for component declaration marker inference.

Run with ``uv run pyright``.
"""

import cimba.sim as sim


class Supplier(sim.Component):
    priority: sim.Const[int]


class Consumer(sim.Component):
    count: sim.Const[int]
    supplier: sim.Ref[Supplier]
    suppliers: sim.Refs[Supplier]

    def reorder(self) -> int:
        supplier: Supplier = self.supplier
        first: Supplier = self.suppliers[0]
        return self.count + supplier.priority + first.priority


class Configurable(sim.Component):
    lot_size: sim.Const[float] = 100.0


configured = Configurable(lot_size=250)
assert configured.lot_size == 250


class DirectRefs(sim.Component):
    source: sim.Ref[Supplier]
    routes: sim.Refs[Supplier]


direct_refs = DirectRefs(
    source=Supplier(priority=3),
    routes=(Supplier(priority=4), Supplier(priority=5)),
)
