import numpy as np

from tutorial import multi_echelon_inventory as inv


def _single_node_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([[10.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.zeros(inv.NUM_NODES, dtype=np.float64),
    )


def _single_node_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([10000.0, 20.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 20.0, 0.0, 0.0, 0.0, 0.0]),
    )


def _single_node_experiment(*, backorder: bool):
    demand, lead_time_delay, base_lead_time = _single_node_data()
    base_stock, reorder_point, initial_inventory = _single_node_parameters()
    demand_by_facility = np.zeros((inv.NUM_NODES, demand.shape[0]))
    demand_by_facility[1:, :] = demand.T

    return inv.model.experiment(
        backorder=1.0 if backorder else 0.0,
        base_stock=base_stock,
        reorder_point=reorder_point,
        initial_inventory=initial_inventory,
        base_lead_time=base_lead_time,
        lead_time_delay=lead_time_delay,
        facilities__demand=demand_by_facility,
        replications=1,
        duration=4.0,
        warmup=0.0,
        seed=1,
    )


def test_lost_sales_policy_runs_from_manual_parameters():
    exp = _single_node_experiment(backorder=False)

    assert exp.run() == 0
    assert exp["facilities__avg_on_hand"].shape == (1, inv.NUM_NODES)
    assert np.isclose(exp["facilities__avg_on_hand"][0, 1], 7.5)
    assert np.isclose(
        exp["facilities__service_level"][0, 1],
        20.0 / (30.0 + inv.EPSILON),
    )
    assert exp["facilities__service_level"][0, 0] == 1.0


def test_backorder_policy_uses_late_sales_fill_rate():
    exp = _single_node_experiment(backorder=True)

    assert exp.run() == 0
    assert np.isclose(exp.trials["facilities__total_late_sales"][0, 1], 10.0)
    assert np.isclose(exp.trials["facilities__backorder"][0, 1], 10.0)
    assert np.isclose(
        exp["facilities__service_level"][0, 1],
        1.0 - 10.0 / (30.0 + inv.EPSILON),
    )


def test_inventory_and_service_outputs_can_be_aggregated():
    exp = _single_node_experiment(backorder=False)
    assert exp.run() == 0

    avg_on_hand = exp["facilities__avg_on_hand"].mean(axis=0)
    service_level = exp["facilities__service_level"].mean(axis=0)

    expected_service = 20.0 / (30.0 + inv.EPSILON)
    assert np.isclose(avg_on_hand[1:].sum(), 7.5)
    assert np.isclose(service_level[1], expected_service)


def test_load_data_reads_csv_layout(tmp_path):
    (tmp_path / "demandData.csv").write_text(
        "0,1,2,3,4\n"
        "1.0,2.0,0.0,3.0,4.0\n"
        "5.0,6.0,0.0,7.0,8.0\n",
        encoding="utf-8",
    )
    (tmp_path / "leadTimeExtraDays.csv").write_text(
        "0\n2\n",
        encoding="utf-8",
    )

    demand, lead_time_delay = inv.load_data(tmp_path)

    assert demand.shape == (2, inv.STOCKING_NODES)
    assert lead_time_delay.tolist() == [0.0, 2.0]


def test_bundled_data_is_available():
    demand, lead_time_delay = inv.load_data()

    assert demand.shape == (10000, inv.STOCKING_NODES)
    assert lead_time_delay.shape == (10000,)
