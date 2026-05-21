import cimba


def test_data_summary_dataset_and_timeseries_match_native_semantics():
    summary = cimba.DataSummary()
    for value in (1.0, 2.0, 3.0):
        summary.add(value)

    assert summary.count == 3
    assert summary.min == 1.0
    assert summary.max == 3.0
    assert summary.mean == 2.0
    assert summary.variance == 1.0
    assert summary.stddev == 1.0

    weighted = cimba.WeightedSummary()
    weighted.add(0.0, 1.0)
    weighted.add(10.0, 3.0)
    assert weighted.count == 2
    assert weighted.weight_sum == 4.0
    assert weighted.mean == 7.5

    dataset = cimba.Dataset()
    for value in (3.0, 1.0, 2.0):
        dataset.add(value)
    assert dataset.values() == [3.0, 1.0, 2.0]
    assert dataset.median == 2.0
    assert dataset.summary().mean == 2.0

    series = cimba.TimeSeries()
    series.add(0.0, 0.0)
    series.add(2.0, 5.0)
    series.finalize(10.0)
    assert series.values() == [(0.0, 0.0, 5.0), (5.0, 2.0, 5.0), (10.0, 2.0, 0.0)]
    assert series.summary().mean == 1.0
