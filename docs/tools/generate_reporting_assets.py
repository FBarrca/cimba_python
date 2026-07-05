"""Generate reporting tutorial artifacts from the public cimba.reporting API.

Run from the repository root with:

    python docs/tools/generate_reporting_assets.py

The text report only needs Cimba. The SVG plots require the optional plotting
extra, for example `pip install -e ".[plot]"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cimba
from cimba import reporting

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "docs" / "static"
REPORT_TEXT = STATIC / "mm1_reporting_report.txt"
FIVE_NUMBER_TEXT = STATIC / "mm1_reporting_five_number.txt"
HISTOGRAM_SVG = STATIC / "mm1_reporting_histogram.svg"
PACF_SVG = STATIC / "mm1_reporting_pacf.svg"


@dataclass
class MM1Trial:
    arr_rate: float = 0.75
    srv_rate: float = 1.0
    warmup_time: float = 1000.0
    duration: float = 1_000_000.0
    seed: int = 14


def arrival(ctx: MM1Trial) -> None:
    while True:
        cimba.hold(cimba.random.exponential(1.0 / ctx.arr_rate))
        ctx.queue.put(1)


def service(ctx: MM1Trial) -> None:
    while True:
        ctx.queue.get(1)
        cimba.hold(cimba.random.exponential(1.0 / ctx.srv_rate))


def recorder(ctx: MM1Trial) -> None:
    cimba.hold(ctx.warmup_time)
    ctx.queue.start_recording()
    cimba.hold(ctx.duration)
    ctx.queue.stop_recording()
    ctx.arrival_process.stop()
    ctx.service_process.stop()
    ctx.simulation.clear()


def make_reporting_inputs() -> tuple[cimba.TimeSeries, reporting.HistoryReport]:
    trial = MM1Trial()
    with cimba.Simulation(seed=trial.seed) as sim:
        trial.simulation = sim
        trial.queue = cimba.Buffer("Queue")
        trial.arrival_process = cimba.Process("Arrival", arrival, trial).start()
        trial.service_process = cimba.Process("Service", service, trial).start()
        cimba.Process("Recorder", recorder, trial).start()
        sim.execute()
        history = trial.queue.history().copy()
        report = reporting.resource_report(trial.queue, lags=20, correlation="pacf")
        return history, report


def write_text_report(report: reporting.HistoryReport) -> None:
    REPORT_TEXT.write_text(reporting.format_report(report) + "\n", encoding="utf-8")


def write_five_number(history: cimba.TimeSeries) -> None:
    five = reporting.five_number(history)
    FIVE_NUMBER_TEXT.write_text(
        (
            "FiveNumberSummary("
            f"min={five.min:.6g}, "
            f"q1={five.q1:.6g}, "
            f"median={five.median:.6g}, "
            f"q3={five.q3:.6g}, "
            f"max={five.max:.6g}, "
            f"weighted={five.weighted})\n"
        ),
        encoding="utf-8",
    )


def write_plot_reports(report: reporting.HistoryReport) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit(
            "Install the optional plotting dependency with `pip install -e \".[plot]\"` "
            "before regenerating the tutorial SVGs."
        ) from exc

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
    reporting.plot_histogram(report.histogram, ax=ax, color="#2f7f95")
    ax.set_title("M/M/1 queue level histogram")
    ax.tick_params(axis="x", labelsize=7)
    fig.savefig(HISTOGRAM_SVG, format="svg", metadata={"Date": None})
    plt.close(fig)

    if report.correlogram is None:
        raise ValueError("expected a correlogram in the generated report")

    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    reporting.plot_correlogram(report.correlogram, ax=ax, color="#2f7d54")
    ax.set_title("M/M/1 queue partial autocorrelation")
    ax.set_ylim(-0.25, 1.05)
    fig.savefig(PACF_SVG, format="svg", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    history, report = make_reporting_inputs()
    write_five_number(history)
    write_text_report(report)
    write_plot_reports(report)


if __name__ == "__main__":
    main()
