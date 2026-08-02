"""Pyright fixture for spawnable component-process inference.

Run with ``uv run pyright``.
"""

import cimba.sim as sim


class Shipment(sim.Struct):
    amount: int


class Policy(sim.Component):
    @sim.process(spawnable=True)
    def allocate(self, env, shipment: Shipment):
        sim.hold(0.0)


class Material(sim.Component):
    policy: Policy = Policy()

    @sim.process(spawnable=True)
    def replenishment(self, env, shipment: Shipment):
        sim.hold(0.0)

    @sim.process
    def reorder(self, env):
        direct = sim.spawn(self.replenishment, env)
        nested = sim.spawn(self.policy.allocate, env)
        Shipment(direct)
        Shipment(nested)

    @sim.process
    def ordinary(self, env):
        sim.hold(0.0)

    @sim.function
    def helper(self, env):
        return 0

    @sim.process
    def rejected(self, env):
        # These must remain static errors: only spawnable process descriptors
        # are accepted by sim.spawn.
        sim.spawn(self.ordinary, env)  # type: ignore[reportArgumentType]
        sim.spawn(self.helper, env)  # type: ignore[reportArgumentType]
