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


def test_data_helpers_copy_merge_reset_sort_and_correlations():
    left = cimba.Dataset()
    right = cimba.Dataset()
    for value in (3.0, 1.0):
        left.add(value)
    for value in (2.0, 4.0):
        right.add(value)

    copied = left.copy()
    assert copied.values() == [3.0, 1.0]
    merged = left.merge(right)
    assert merged.values() == [3.0, 1.0, 2.0, 4.0]
    merged.sort()
    assert merged.values() == [1.0, 2.0, 3.0, 4.0]
    assert len(merged.acf(2)) == 3
    assert len(merged.pacf(2)) == 3
    merged.reset()
    assert merged.count == 0

    summary = cimba.DataSummary()
    summary.add(1.0)
    other_summary = cimba.DataSummary()
    other_summary.add(3.0)
    assert summary.merge(other_summary).mean == 2.0
    summary.reset()
    assert summary.count == 0

    weighted = cimba.WeightedSummary()
    weighted.add(1.0, 1.0)
    other_weighted = cimba.WeightedSummary()
    other_weighted.add(3.0, 1.0)
    assert weighted.merge(other_weighted).mean == 2.0
    weighted.reset()
    assert weighted.count == 0

    series = cimba.TimeSeries()
    series.add(2.0, 1.0)
    series.add(1.0, 2.0)
    series.add(3.0, 3.0)
    copied_series = series.copy()
    assert copied_series.values() == [(1.0, 2.0, 1.0), (2.0, 1.0, 1.0), (3.0, 3.0, 0.0)]
    series.sort_by_value()
    assert [row[1] for row in series.values()] == [1.0, 2.0, 3.0]
    series.sort_by_time()
    assert [row[0] for row in series.values()] == [1.0, 2.0, 3.0]
    assert len(series.acf(1)) == 2
    assert len(series.pacf(1)) == 2
    series.reset()
    assert series.count == 0
