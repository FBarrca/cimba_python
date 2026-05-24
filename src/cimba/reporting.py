"""Structured reporting and optional plotting helpers for Cimba data."""

from __future__ import annotations

import builtins
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .cmb_buffer import UNLIMITED, Buffer
from .cmb_datasummary import DataSummary
from .cmb_dataset import Dataset
from .cmb_objectqueue import ObjectQueue
from .cmb_priorityqueue import PriorityQueue
from .cmb_resource import Resource
from .cmb_resourcepool import ResourcePool
from .cmb_timeseries import TimeSeries
from .cmb_wtdsummary import WeightedSummary

CorrelationKind = Literal["acf", "pacf"]
WeightedMode = Literal["auto"] | bool


@dataclass(frozen=True)
class SummaryStats:
    """Availability-aware summary statistics."""

    count: int
    weight_sum: float | None
    min: float | None
    max: float | None
    mean: float | None
    variance: float | None
    stddev: float | None
    skewness: float | None
    kurtosis: float | None


@dataclass(frozen=True)
class FiveNumberSummary:
    """Five-number summary for sample values."""

    min: float
    q1: float
    median: float
    q3: float
    max: float
    weighted: bool = False


@dataclass(frozen=True)
class HistogramBin:
    """One histogram bucket."""

    lower: float | None
    upper: float | None
    mass: float
    underflow: bool = False
    overflow: bool = False


@dataclass(frozen=True)
class Histogram:
    """Histogram data with explicit underflow and overflow buckets."""

    bins: tuple[HistogramBin, ...]
    total_mass: float
    weighted: bool
    range: tuple[float, float]


@dataclass(frozen=True)
class Correlogram:
    """Autocorrelation or partial-autocorrelation coefficients."""

    kind: CorrelationKind
    lags: tuple[int, ...]
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class HistoryReport:
    """Structured report for a recorded history or dataset."""

    title: str
    summary: SummaryStats
    histogram: Histogram
    correlogram: Correlogram | None = None


def summarize(source: DataSummary | WeightedSummary | Dataset | TimeSeries) -> SummaryStats:
    """Return availability-aware summary statistics for a Cimba data object."""

    if isinstance(source, Dataset):
        if source.count == 0:
            raise ValueError("cannot summarize an empty dataset")
        return summarize(source.summary())

    if isinstance(source, TimeSeries):
        _weighted_time_series_samples(source)
        return summarize(source.summary())

    if isinstance(source, WeightedSummary):
        return _summary_from_native(source, weight_sum=source.weight_sum)

    if isinstance(source, DataSummary):
        return _summary_from_native(source, weight_sum=None)

    raise TypeError("source must be a DataSummary, WeightedSummary, Dataset, or TimeSeries")


def histogram(
    source: Dataset | TimeSeries,
    bins: int = 20,
    range: tuple[float, float] | None = None,
    weighted: WeightedMode = "auto",
) -> Histogram:
    """Build a histogram from a Dataset or TimeSeries."""

    if bins <= 0:
        raise ValueError("bins must be positive")

    values, weights, is_weighted = _histogram_samples(source, weighted)
    if not values:
        if isinstance(source, Dataset):
            raise ValueError("cannot histogram an empty dataset")
        raise ValueError("cannot histogram a time series without intervals")

    low, high, finite_bins = _histogram_bounds(values, bins, range)
    masses = [0.0] * (finite_bins + 2)
    width = (high - low) / finite_bins

    for value, mass in zip(values, weights, strict=True):
        if value < low:
            index = 0
        elif value >= high:
            index = finite_bins + 1
        else:
            index = 1 + int((value - low) / width)
            if index > finite_bins:
                index = finite_bins
        masses[index] += mass

    result_bins: list[HistogramBin] = [
        HistogramBin(None, low, masses[0], underflow=True)
    ]
    for index in builtins.range(finite_bins):
        lower = low + index * width
        result_bins.append(HistogramBin(lower, lower + width, masses[index + 1]))
    result_bins.append(HistogramBin(high, None, masses[-1], overflow=True))

    return Histogram(
        bins=tuple(result_bins),
        total_mass=sum(weights),
        weighted=is_weighted,
        range=(low, high),
    )


def five_number(source: Dataset | TimeSeries) -> FiveNumberSummary:
    """Return a five-number summary for a dataset or time series."""

    if isinstance(source, Dataset):
        values = sorted(float(value) for value in source.values())
        if not values:
            raise ValueError("cannot calculate five-number summary for an empty dataset")
        return FiveNumberSummary(
            min=values[0],
            q1=_unweighted_quartile(values, 0.25),
            median=_unweighted_quartile(values, 0.50),
            q3=_unweighted_quartile(values, 0.75),
            max=values[-1],
            weighted=False,
        )

    if isinstance(source, TimeSeries):
        values, weights = _weighted_time_series_samples(source)
        pairs = sorted(
            (value, weight)
            for value, weight in zip(values, weights, strict=True)
            if weight > 0.0
        )
        if not pairs:
            raise ValueError("time series needs at least one positive-duration interval")
        sorted_values = [pair[0] for pair in pairs]
        sorted_weights = [pair[1] for pair in pairs]
        return FiveNumberSummary(
            min=sorted_values[0],
            q1=_weighted_quantile(sorted_values, sorted_weights, 0.25),
            median=_weighted_quantile(sorted_values, sorted_weights, 0.50),
            q3=_weighted_quantile(sorted_values, sorted_weights, 0.75),
            max=sorted_values[-1],
            weighted=True,
        )

    raise TypeError("source must be a Dataset or TimeSeries")


def correlogram(
    source: Dataset | TimeSeries,
    lags: int,
    kind: CorrelationKind = "acf",
) -> Correlogram:
    """Calculate autocorrelation or partial-autocorrelation coefficients."""

    if kind == "acf":
        coefficients = source.acf(lags)
    elif kind == "pacf":
        coefficients = source.pacf(lags)
    else:
        raise ValueError("kind must be 'acf' or 'pacf'")

    return Correlogram(
        kind=kind,
        lags=tuple(builtins.range(len(coefficients))),
        coefficients=tuple(coefficients),
    )


def history_report(
    source: Dataset | TimeSeries,
    *,
    title: str | None = None,
    bins: int = 20,
    lags: int | None = None,
    correlation: CorrelationKind | None = None,
) -> HistoryReport:
    """Build a structured report for a dataset or time series."""

    if correlation is not None and lags is None:
        raise ValueError("lags must be provided when correlation is requested")

    corr = None
    if lags is not None:
        corr = correlogram(source, lags, correlation or "acf")

    return HistoryReport(
        title=title or _default_report_title(source),
        summary=summarize(source),
        histogram=histogram(source, bins=bins),
        correlogram=corr,
    )


def resource_report(
    resource: Buffer | ObjectQueue | PriorityQueue | Resource | ResourcePool,
    *,
    bins: int | None = None,
    lags: int | None = None,
    correlation: CorrelationKind | None = None,
) -> HistoryReport:
    """Build a structured report for a recorded resource or queue history."""

    title, default_bins, value_range = _resource_report_defaults(resource)
    report_bins = default_bins if bins is None else bins

    if correlation is not None and lags is None:
        raise ValueError("lags must be provided when correlation is requested")

    history = resource.history()
    corr = None
    if lags is not None:
        corr = correlogram(history, lags, correlation or "acf")

    return HistoryReport(
        title=title,
        summary=summarize(history),
        histogram=histogram(history, bins=report_bins, range=value_range),
        correlogram=corr,
    )


def format_report(report: HistoryReport) -> str:
    """Format a structured report as text identical to the native Cimba report.

    Mirrors ``cmb_buffer_print_report()`` (the weighted summary line plus the
    character histogram) followed by ``cmb_dataset_correlogram_print()`` so the
    output matches the C tutorial byte for byte.
    """

    lines = [report.title, format_summary(report.summary)]
    lines.extend(_format_histogram(report.histogram))
    if report.correlogram is not None:
        lines.extend(_format_correlogram(report.correlogram))
    return "\n".join(lines)


def format_summary(
    summary_or_source: SummaryStats
    | DataSummary
    | WeightedSummary
    | Dataset
    | TimeSeries,
) -> str:
    """Format summary statistics as compact debug text."""

    stats = (
        summary_or_source
        if isinstance(summary_or_source, SummaryStats)
        else summarize(summary_or_source)
    )
    return _format_summary(stats)


def plot_history(history: TimeSeries, ax=None, **kwargs):
    """Plot a time series history and return the Matplotlib axes."""

    plt = _load_pyplot()
    if ax is None:
        _, ax = plt.subplots()

    rows = history.values()
    if not rows:
        raise ValueError("cannot plot an empty time series")

    times = [row[0] for row in rows]
    values = [row[1] for row in rows]
    kwargs.setdefault("drawstyle", "steps-post")
    ax.plot(times, values, **kwargs)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    return ax


def plot_histogram(histogram_or_source: Histogram | Dataset | TimeSeries, ax=None, **kwargs):
    """Plot a Histogram, Dataset, or TimeSeries and return the Matplotlib axes."""

    plt = _load_pyplot()
    if ax is None:
        _, ax = plt.subplots()

    hist = (
        histogram_or_source
        if isinstance(histogram_or_source, Histogram)
        else histogram(histogram_or_source)
    )
    x = list(builtins.range(len(hist.bins)))
    ax.bar(x, [bin.mass for bin in hist.bins], **kwargs)
    ax.set_xticks(x)
    ax.set_xticklabels([_bin_label(bin) for bin in hist.bins], rotation=45, ha="right")
    ax.set_ylabel("weighted mass" if hist.weighted else "count")
    ax.set_xlabel("value")
    return ax


def plot_correlogram(
    correlogram_or_source: Correlogram | Dataset | TimeSeries,
    ax=None,
    **kwargs,
):
    """Plot a Correlogram or calculate one from a source with ``lags=...``."""

    plt = _load_pyplot()
    if ax is None:
        _, ax = plt.subplots()

    if isinstance(correlogram_or_source, Correlogram):
        corr = correlogram_or_source
    else:
        lags = kwargs.pop("lags", None)
        kind = kwargs.pop("kind", "acf")
        if lags is None:
            raise TypeError("lags must be provided when plotting a source")
        corr = correlogram(correlogram_or_source, lags, kind)

    # Skip the trivial lag-0 coefficient (always 1.0) so the plot starts at
    # lag 1, matching the native correlogram and format_report() text output.
    lags = [lag for lag in corr.lags if lag != 0]
    coefficients = [
        coefficient
        for lag, coefficient in zip(corr.lags, corr.coefficients, strict=True)
        if lag != 0
    ]

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.bar(lags, coefficients, **kwargs)
    ax.set_xlabel("lag")
    ax.set_ylabel(corr.kind)
    return ax


def plot_report(report: HistoryReport, axes: Sequence[object] | None = None, **kwargs):
    """Plot report summary text, histogram, and optional correlogram."""

    plt = _load_pyplot()
    plot_count = 3 if report.correlogram is not None else 2
    if axes is None:
        figsize = kwargs.pop("figsize", (8, 3 * plot_count))
        _, axes = plt.subplots(plot_count, 1, figsize=figsize)

    axes_list = _axes_list(axes)
    if len(axes_list) < plot_count:
        raise ValueError(f"expected at least {plot_count} axes")

    summary_ax = axes_list[0]
    summary_ax.axis("off")
    summary_ax.set_title(report.title)
    summary_ax.text(
        0.0,
        1.0,
        format_summary(report.summary),
        transform=summary_ax.transAxes,
        va="top",
        family="monospace",
    )
    plot_histogram(report.histogram, ax=axes_list[1], **kwargs)
    if report.correlogram is not None:
        plot_correlogram(report.correlogram, ax=axes_list[2], **kwargs)
    return tuple(axes_list[:plot_count])


def _summary_from_native(
    summary: DataSummary | WeightedSummary,
    *,
    weight_sum: float | None,
) -> SummaryStats:
    count = int(summary.count)
    return SummaryStats(
        count=count,
        weight_sum=weight_sum if weight_sum is not None and count > 0 else None,
        min=summary.min if count > 0 else None,
        max=summary.max if count > 0 else None,
        mean=summary.mean if count > 0 else None,
        variance=summary.variance if count > 1 else None,
        stddev=summary.stddev if count > 1 else None,
        skewness=summary.skewness if count > 2 else None,
        kurtosis=summary.kurtosis if count > 3 else None,
    )


def _histogram_samples(
    source: Dataset | TimeSeries,
    weighted: WeightedMode,
) -> tuple[list[float], list[float], bool]:
    if isinstance(source, Dataset):
        if weighted is True:
            raise ValueError("Dataset histograms cannot be weighted")
        values = [float(value) for value in source.values()]
        return values, [1.0] * len(values), False

    if isinstance(source, TimeSeries):
        if weighted == "auto" or weighted is True:
            values, weights = _weighted_time_series_samples(source)
            return values, weights, True

        rows = source.values()
        values = [float(row[1]) for row in rows]
        return values, [1.0] * len(values), False

    raise TypeError("source must be a Dataset or TimeSeries")


def _weighted_time_series_samples(source: TimeSeries) -> tuple[list[float], list[float]]:
    rows = source.values()
    if len(rows) < 2:
        raise ValueError("time series needs at least one completed interval")

    values = [float(row[1]) for row in rows[:-1]]
    weights = [float(row[2]) for row in rows[:-1]]
    if not values or sum(weights) <= 0.0:
        raise ValueError("time series needs at least one positive-duration interval")
    return values, weights


def _histogram_bounds(
    values: list[float],
    requested_bins: int,
    value_range: tuple[float, float] | None,
) -> tuple[float, float, int]:
    if value_range is None:
        low = min(values)
        high = max(values)
    else:
        low, high = value_range

    low = float(low)
    high = float(high)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("histogram range must be finite")
    if high < low:
        raise ValueError("histogram range upper bound must be >= lower bound")

    if high == low:
        center = low
        low = center - 0.5
        high = center + 0.5
        return low, high, 1

    return low, high, requested_bins


def _unweighted_quartile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]

    if fraction == 0.50:
        return _median(values)

    midpoint = len(values) // 2
    if fraction == 0.25:
        sample = values[:midpoint]
    elif len(values) % 2 == 0:
        sample = values[midpoint:]
    else:
        sample = values[midpoint + 1 :]

    return _median(sample) if sample else values[0]


def _median(values: Sequence[float]) -> float:
    midpoint = len(values) // 2
    if len(values) % 2 == 0:
        return (values[midpoint - 1] + values[midpoint]) / 2.0
    return values[midpoint]


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    fraction: float,
) -> float:
    if len(values) == 1:
        return values[0]

    target = fraction * sum(weights)
    cumulative = 0.0
    previous_cumulative = 0.0
    previous_value = values[0]

    for value, weight in zip(values, weights, strict=True):
        cumulative += weight
        if target <= cumulative:
            if cumulative == previous_cumulative:
                return value
            if previous_cumulative == 0.0:
                return previous_value
            position = (target - previous_cumulative) / (cumulative - previous_cumulative)
            return previous_value + (value - previous_value) * position
        previous_cumulative = cumulative
        previous_value = value

    return values[-1]


def _resource_report_defaults(
    resource: Buffer | ObjectQueue | PriorityQueue | Resource | ResourcePool,
) -> tuple[str, int, tuple[float, float] | None]:
    if isinstance(resource, Buffer):
        return (
            f"Buffer levels for {resource.name}",
            _finite_capacity_bins(resource.capacity),
            _finite_capacity_range(resource.capacity),
        )
    if isinstance(resource, ObjectQueue):
        return (
            f"Queue lengths for {resource.name}",
            _finite_capacity_bins(resource.capacity),
            _finite_capacity_range(resource.capacity),
        )
    if isinstance(resource, PriorityQueue):
        return (
            f"Queue lengths for {resource.name}",
            _finite_capacity_bins(resource.capacity),
            _finite_capacity_range(resource.capacity),
        )
    if isinstance(resource, Resource):
        return f"Resource utilization for {resource.name}", 2, (0.0, 2.0)
    if isinstance(resource, ResourcePool):
        return (
            f"Pool resource utilization for {resource.name}",
            _finite_capacity_bins(resource.capacity),
            _finite_capacity_range(resource.capacity),
        )
    raise TypeError("resource must be a Buffer, queue, Resource, or ResourcePool")


def _finite_capacity_bins(capacity: int) -> int:
    if capacity == UNLIMITED:
        return 20
    return min(20, int(capacity) + 1)


def _finite_capacity_range(capacity: int) -> tuple[float, float] | None:
    if capacity == UNLIMITED:
        return None
    return 0.0, float(capacity) + 1.0


def _default_report_title(source: Dataset | TimeSeries) -> str:
    if isinstance(source, Dataset):
        return "Dataset report"
    return "Time series report"


def _format_summary(summary: SummaryStats) -> str:
    """Format a summary line like the native ``cmb_datasummary_print`` lead-ins.

    The native printer omits the weight sum and the min/max, and emits each
    moment with ``%#8.4g`` only once the sample is large enough to define it.
    """

    parts = [f"N {summary.count:8d}"]
    for label, value in (
        ("Mean", summary.mean),
        ("StdDev", summary.stddev),
        ("Variance", summary.variance),
        ("Skewness", summary.skewness),
        ("Kurtosis", summary.kurtosis),
    ):
        if value is not None:
            parts.append(f"  {label} {value:#8.4g}")
    return "".join(parts)


def _format_histogram(hist: Histogram) -> list[str]:
    """Render a character histogram like ``cmi_dataset_histogram_print``."""

    separator = "-" * 80
    binmax = max((bin.mass for bin in hist.bins), default=0.0)
    scale = binmax / 50.0 if binmax > 0.0 else 1.0
    low_lim, high_lim = hist.range

    lines = [separator]
    lines.append(
        f"( -Infinity, {low_lim:#10.4g})   |{_histogram_blocks(hist.bins[0].mass, scale)}"
    )
    for bin in hist.bins[1:-1]:
        lines.append(
            f"[{bin.lower:#10.4g}, {bin.upper:#10.4g})   |"
            f"{_histogram_blocks(bin.mass, scale)}"
        )
    lines.append(
        f"[{high_lim:#10.4g},  Infinity )   |{_histogram_blocks(hist.bins[-1].mass, scale)}"
    )
    lines.append(separator)
    return lines


def _histogram_blocks(mass: float, scale: float) -> str:
    """One histogram bar: ``#`` per full unit, ``=`` past half, ``-`` for any."""

    filled = int(mass / scale)
    remainder = mass / scale - filled
    blocks = "#" * filled
    if remainder >= 0.5:
        blocks += "="
    elif remainder > 0.0:
        blocks += "-"
    return blocks


def _format_correlogram(correlogram: Correlogram) -> list[str]:
    """Render a correlogram like ``cmb_dataset_correlogram_print`` (lags 1..n)."""

    line_length = 80
    max_bar_width = (line_length - 14) // 2
    gap = " " * (max_bar_width - 3)
    lines = [f"{' ' * 11}-1.0{gap}0.0{gap}1.0", "-" * line_length]
    for lag, coefficient in zip(
        correlogram.lags, correlogram.coefficients, strict=True
    ):
        if lag == 0:
            continue
        lines.append(
            f"{lag:4d}  {coefficient:#6.3f} {_correlogram_bar(coefficient, max_bar_width)}"
        )
    lines.append("-" * line_length)
    return lines


def _correlogram_bar(value: float, max_bar_width: int) -> str:
    """One correlogram bar drawn either side of the centre axis."""

    value = max(-1.0, min(1.0, value))
    bar_width = max_bar_width * abs(value)
    filled = int(math.floor(bar_width))
    remainder = bar_width - filled

    if value < 0.0:
        spaces = max(max_bar_width - filled - 1, 0)
        mark = "=" if remainder > 0.5 else "-" if remainder > 0.0 else " "
        return f"{' ' * spaces}{mark}{'#' * filled}|"

    tail = "=" if remainder > 0.5 else "-" if remainder > 0.0 else ""
    return f"{' ' * max_bar_width}|{'#' * filled}{tail}"


def _bin_label(bin: HistogramBin) -> str:
    if bin.underflow:
        return f"(-inf, {_fmt(bin.upper)})"
    if bin.overflow:
        return f"[{_fmt(bin.lower)}, inf)"
    return f"[{_fmt(bin.lower)}, {_fmt(bin.upper)})"


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _axes_list(axes: Sequence[object] | object) -> list[object]:
    if hasattr(axes, "ravel"):
        return list(axes.ravel())
    if isinstance(axes, Sequence):
        return list(axes)
    return [axes]


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting helpers require Matplotlib. Install the optional "
            "dependency with 'cimba[plot]'."
        ) from exc
    return plt


__all__ = [
    "Correlogram",
    "FiveNumberSummary",
    "HistoryReport",
    "Histogram",
    "HistogramBin",
    "SummaryStats",
    "correlogram",
    "five_number",
    "format_report",
    "format_summary",
    "histogram",
    "history_report",
    "plot_correlogram",
    "plot_histogram",
    "plot_history",
    "plot_report",
    "resource_report",
    "summarize",
]
