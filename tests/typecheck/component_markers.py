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
