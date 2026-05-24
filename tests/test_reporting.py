import builtins

import pytest

import cimba
from cimba import reporting


def test_summarize_reports_available_statistics_without_native_sentinels():
    summary = cimba.DataSummary()
    empty = reporting.summarize(summary)
    assert empty.count == 0
    assert empty.min is None
    assert empty.mean is None
    assert empty.variance is None

    summary.add(2.0)
    one = reporting.summarize(summary)
    assert one.count == 1
    assert one.min == 2.0
    assert one.max == 2.0
    assert one.mean == 2.0
    assert one.variance is None

    summary.add(4.0)
    two = reporting.summarize(summary)
    assert two.variance is not None
    assert two.stddev is not None
    assert two.skewness is None

    weighted = cimba.WeightedSummary()
    weighted.add(1.0, 2.0)
    weighted.add(3.0, 2.0)
    weighted_stats = reporting.summarize(weighted)
    assert weighted_stats.weight_sum == 4.0
    assert weighted_stats.mean == 2.0


def test_format_summary_is_pythonic_datasummary_print_equivalent():
    summary = cimba.DataSummary()
    for value in (1.0, 2.0, 3.0, 4.0):
        summary.add(value)

    text = reporting.format_summary(summary)

    assert text.startswith("N        4")
    assert "Mean" in text
    assert "StdDev" in text
    assert reporting.format_summary(reporting.summarize(summary)) == text


def test_summarize_rejects_empty_dataset_and_incomplete_time_series():
    dataset = cimba.Dataset()
    with pytest.raises(ValueError, match="empty dataset"):
        reporting.summarize(dataset)

    series = cimba.TimeSeries()
    series.add(1.0, 0.0)
    with pytest.raises(ValueError, match="completed interval"):
        reporting.summarize(series)


def test_histogram_unweighted_dataset_with_overflow_bins():
    dataset = cimba.Dataset()
    for value in (-1.0, 0.0, 0.25, 0.75, 1.5):
        dataset.add(value)

    hist = reporting.histogram(dataset, bins=2, range=(0.0, 1.0))

    assert hist.weighted is False
    assert hist.total_mass == 5.0
    assert [bin.mass for bin in hist.bins] == [1.0, 2.0, 1.0, 1.0]
    assert hist.bins[0].underflow is True
    assert hist.bins[-1].overflow is True


def test_histogram_time_series_uses_duration_weights():
    series = cimba.TimeSeries()
    series.add(0.0, 0.0)
    series.add(2.0, 5.0)
    series.add(4.0, 7.0)

    hist = reporting.histogram(series, bins=2, range=(0.0, 4.0))

    assert hist.weighted is True
    assert hist.total_mass == 7.0
    assert [bin.mass for bin in hist.bins] == [0.0, 5.0, 2.0, 0.0]


def test_histogram_constant_values_and_invalid_range():
    dataset = cimba.Dataset()
    dataset.add(3.0)
    dataset.add(3.0)

    hist = reporting.histogram(dataset, bins=20)

    assert hist.range == (2.5, 3.5)
    assert [bin.mass for bin in hist.bins] == [0.0, 2.0, 0.0]

    with pytest.raises(ValueError, match="upper bound"):
        reporting.histogram(dataset, range=(2.0, 1.0))


def test_five_number_dataset_uses_cimba_quartile_convention():
    dataset = cimba.Dataset()
    for value in (5.0, 1.0, 4.0, 2.0, 3.0):
        dataset.add(value)

    five = reporting.five_number(dataset)

    assert five == reporting.FiveNumberSummary(
        min=1.0,
        q1=1.5,
        median=3.0,
        q3=4.5,
        max=5.0,
        weighted=False,
    )


def test_five_number_time_series_uses_duration_weights():
    series = cimba.TimeSeries()
    series.add(0.0, 0.0)
    series.add(10.0, 2.0)
    series.add(20.0, 6.0)
    series.add(20.0, 7.0)

    five = reporting.five_number(series)

    assert five.min == 0.0
    assert five.q1 == pytest.approx(0.0)
    assert five.median == pytest.approx(3.75)
    assert five.q3 == pytest.approx(8.125)
    assert five.max == 20.0
    assert five.weighted is True


def test_correlogram_delegates_to_native_acf_and_pacf():
    dataset = cimba.Dataset()
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        dataset.add(value)

    acf = reporting.correlogram(dataset, 2, "acf")
    pacf = reporting.correlogram(dataset, 2, "pacf")

    assert acf.kind == "acf"
    assert acf.lags == (0, 1, 2)
    assert acf.coefficients == tuple(dataset.acf(2))
    assert pacf.kind == "pacf"
    assert pacf.coefficients == tuple(dataset.pacf(2))

    with pytest.raises(ValueError, match="kind"):
        reporting.correlogram(dataset, 1, "bogus")  # type: ignore[arg-type]


def test_resource_report_uses_recorded_buffer_history_defaults():
    def actor(buffer):
        assert buffer.put(2) == (cimba.SUCCESS, 0)
        cimba.hold(1.0)
        assert buffer.get(1) == (cimba.SUCCESS, 1)
        cimba.hold(1.0)

    with cimba.Simulation(seed=1) as sim:
        buffer = cimba.Buffer("Buf", capacity=3)
        buffer.start_recording()
        cimba.Process("Actor", actor, buffer).start()
        sim.execute()
        buffer.stop_recording()
        report = reporting.resource_report(buffer)

    assert report.title == "Buffer levels for Buf"
    assert report.summary.count > 0
    assert report.histogram.weighted is True
    assert report.histogram.range == (0.0, 4.0)


def test_history_report_and_format_report_include_optional_correlogram():
    dataset = cimba.Dataset()
    for value in (1.0, 2.0, 3.0, 4.0):
        dataset.add(value)

    report = reporting.history_report(dataset, title="Samples", bins=3, lags=1)
    text = reporting.format_report(report)
    lines = text.splitlines()

    assert report.title == "Samples"
    assert report.correlogram is not None
    # Native-style layout: title, summary line, then the character histogram
    # and correlogram blocks bracketed by 80-character separator lines.
    assert lines[0] == "Samples"
    assert lines[1].startswith("N ")
    assert "( -Infinity," in text
    assert "Infinity )" in text
    assert "-1.0" in text and "1.0" in text
    assert "-" * 80 in lines


def test_reporting_import_is_lightweight_and_plotting_error_is_helpful(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    hist = reporting.Histogram(
        bins=(reporting.HistogramBin(0.0, 1.0, 1.0),),
        total_mass=1.0,
        weighted=False,
        range=(0.0, 1.0),
    )
    with pytest.raises(ImportError, match="cimba\\[plot\\]"):
        reporting.plot_histogram(hist)


def test_plot_helpers_with_matplotlib_agg_backend():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    hist = reporting.Histogram(
        bins=(reporting.HistogramBin(0.0, 1.0, 1.0),),
        total_mass=1.0,
        weighted=False,
        range=(0.0, 1.0),
    )
    ax = reporting.plot_histogram(hist)
    assert ax.get_ylabel() == "count"
    plt.close(ax.figure)
