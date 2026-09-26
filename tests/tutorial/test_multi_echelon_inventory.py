import numpy as np

import cimba as cb
from cimba import inputs
from tutorial.multi_echelon_inventory import (
    MultiEchelonInventory, SourceFacility, StockingFacility,
)


def test_joint_demand_can_drive_heterogeneous_facilities():
    model = MultiEchelonInventory()
    assert type(model.facilities[0]) is SourceFacility
    assert all(type(node) is StockingFacility for node in model.facilities[1:])
    demand = np.tile(np.arange(1.0, 6.0), (8, 1))
    joint = inputs.bootstrap.joint(
        {node: demand[:, node - 1] for node in range(1, 6)}, mean_block=2)
    for node in range(1, 6):
        model.facilities[node].demand = joint[node]
    model.lead_time_delay = inputs.trace([0.0], on_exhausted="wrap")
    results = cb.Experiment(
        model, window=cb.Window(duration=5), seed=123,
    ).run()
    assert not results.failed.any()
    assert results[model.facilities[0]].avg_on_hand[0, 0] == 0.0
    assert results[model.facilities[0]].service_level[0, 0] == 1.0
    assert all(results[node].service_level[0, 0] > 0
               for node in model.facilities[1:])
