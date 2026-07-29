import numpy as np

from tutorial import multi_echelon_inventory as inventory


def test_multi_echelon_inventory_uses_concrete_facility_variants():
    model = inventory.model
    schema = model.component_schema("facilities.demand")

    assert type(model.facilities[0]) is inventory.SourceFacility
    assert all(
        type(facility) is inventory.StockingFacility
        for facility in model.facilities[1:]
    )
    assert schema.owners == (1, 2, 3, 4, 5)
    assert schema.packed

    horizon = 8
    exp = model.experiment(
        backorder=0.0,
        base_stock=inventory.BASE_STOCK,
        reorder_point=inventory.REORDER_POINT,
        initial_inventory=inventory.INITIAL_INVENTORY,
        base_lead_time=inventory.BASE_LEAD_TIME,
        lead_time_delay=np.zeros(inventory.STOCKING_NODES * horizon),
        facilities__demand=[
            np.full(horizon, float(node))
            for node in range(1, inventory.NUM_NODES)
        ],
        replications=1,
        duration=5.0,
        warmup=0.0,
        seed=123,
    )

    assert exp.run() == 0
    assert exp["facilities__avg_on_hand"][0, 0] == 0.0
    assert exp["facilities__service_level"][0, 0] == 1.0
    assert np.all(exp["facilities__service_level"][0, 1:] > 0.0)
