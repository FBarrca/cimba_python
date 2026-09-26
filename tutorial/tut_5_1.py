"""Tutorial 5.1: a three-station assembly line with dynamic parts."""

from pathlib import Path

import numpy as np

import cimba as cb
from cimba import inputs
from cimba.diagrams import mermaid


RANDOM_SEED = 45
STATION_NAMES = ("Station 1", "Station 2", "Station 3")
STATION_MEANS = (5.0, 7.0, 4.0)
INTERARRIVAL_TIME = 3.0
SIMULATION_TIME = 10_000.0
PLOT_DIR = Path(__file__).with_name("tut_5_1_plots")


class Part(cb.Model):
    part_id: cb.State[int]
    arrival_system: cb.State[float]
    station_entry: cb.State[float]


class FinishedParts(cb.Model):
    line: cb.Ref["AssemblyLine"]
    inbox: cb.Store[Part]

    def __init__(self, line):
        self.line = line

    @cb.process
    def finish(self):
        while True:
            part = self.inbox.get()
            self.line.cycle_time.record(cb.now() - part.arrival_system)
            self.line.system.get(1)
            cb.release(part)


class Station(cb.Model):
    processing_time: cb.Input[float] = inputs.dist.exponential(mean=5.0)
    inbox: cb.Store[Part]
    downstream: cb.Ref["Station"] | None
    finished: cb.Ref[FinishedParts] | None
    resource: cb.Resource
    wait_time: cb.Dataset
    avg_wait_time: cb.Output[float]
    utilization: cb.Output[float]

    def __init__(self, name: str, mean_processing_time: float, *,
                 downstream=None, finished=None):
        self.name = name
        self.processing_time = inputs.dist.exponential(mean=mean_processing_time)
        self.downstream = downstream
        self.finished = finished

    @cb.process
    def server(self):
        while True:
            part = self.inbox.get()
            self.wait_time.record(cb.now() - part.station_entry)
            self.resource.acquire()
            cb.hold(self.processing_time.next())
            self.resource.release()
            part.station_entry = cb.now()
            if self.downstream is not None:
                self.downstream.inbox.put(part)
            elif self.finished is not None:
                self.finished.inbox.put(part)

    @cb.on_end
    def station_stats(self):
        self.avg_wait_time = self.wait_time.sample_mean()
        self.utilization = 100.0 * self.resource.mean_in_use()


class AssemblyLine(cb.Model):
    duration: cb.Param[float] = SIMULATION_TIME
    arrival_gap: cb.Input[float] = inputs.dist.exponential(mean=INTERARRIVAL_TIME)
    generated_parts: cb.State[int] = 0
    system: cb.Container
    cycle_time: cb.Dataset
    finished_parts: FinishedParts
    station_1: Station
    station_2: Station
    station_3: Station
    total_parts_produced: cb.Output[int]
    avg_cycle_time: cb.Output[float]
    max_cycle_time: cb.Output[float]
    throughput_rate: cb.Output[float]
    avg_number_in_system: cb.Output[float]
    max_number_in_system: cb.Output[float]
    final_number_in_system: cb.Output[int]

    def __init__(self, duration=SIMULATION_TIME):
        self.duration = duration
        self.finished_parts = FinishedParts(self)
        self.station_3 = Station(STATION_NAMES[2], STATION_MEANS[2],
                                 finished=self.finished_parts)
        self.station_2 = Station(STATION_NAMES[1], STATION_MEANS[1],
                                 downstream=self.station_3)
        self.station_1 = Station(STATION_NAMES[0], STATION_MEANS[0],
                                 downstream=self.station_2)
        self.system = cb.Container()
        self.system.capture()
        self.cycle_time = cb.Dataset()
        self.cycle_time.capture()
        for station in (self.station_1, self.station_2, self.station_3):
            station.wait_time = cb.Dataset()
            station.wait_time.capture()

    @cb.process
    def arrivals(self):
        while True:
            cb.hold(self.arrival_gap.next())
            self.generated_parts += 1
            part = cb.spawn(Part, part_id=self.generated_parts,
                            arrival_system=cb.now(), station_entry=cb.now())
            self.system.put(1)
            self.station_1.inbox.put(part)

    @cb.on_end
    def collect_stats(self):
        completed = self.cycle_time.sample_count()
        self.total_parts_produced = completed
        self.avg_cycle_time = self.cycle_time.sample_mean()
        self.max_cycle_time = self.cycle_time.sample_max()
        self.throughput_rate = completed / self.duration
        self.avg_number_in_system = self.system.mean_level()
        self.max_number_in_system = self.system.max_level()
        self.final_number_in_system = self.system.level()


def print_results(model: AssemblyLine, results: cb.Results) -> None:
    line = results[model]
    print("--- Simulation Results Analysis (Cimba) ---")
    print(f"Total parts produced: {int(line.total_parts_produced[0, 0])}")
    print(f"Average cycle time: {line.avg_cycle_time[0, 0]:.2f} minutes")
    print(f"Maximum cycle time: {line.max_cycle_time[0, 0]:.2f} minutes")
    print(f"Throughput rate: {line.throughput_rate[0, 0]:.2f} parts/minute")
    for station in (model.station_1, model.station_2, model.station_3):
        values = results[station]
        print(f"{station.name}: wait {values.avg_wait_time[0, 0]:.2f} min, "
              f"utilization {values.utilization[0, 0]:.2f}%")
    print(f"Average parts in system: {line.avg_number_in_system[0, 0]:.2f}")
    print(f"Maximum parts in system: {line.max_number_in_system[0, 0]:.0f}")
    print(f"Parts still in system: {line.final_number_in_system[0, 0]:.0f}")


def plot_results(model: AssemblyLine, results: cb.Results) -> None:
    try:
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as exc:
        raise SystemExit("Install the plot extra to create figures") from exc
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _, cycle_times = results[model].cycle_time.trial(0, 0)
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(cycle_times, bins=20)
    axes[0].set_title("Part cycle time")
    axes[0].set_xlabel("Minutes")
    utilizations = [results[station].utilization[0, 0] for station in
                    (model.station_1, model.station_2, model.station_3)]
    axes[1].bar(STATION_NAMES, utilizations)
    axes[1].set_title("Station utilization")
    axes[1].set_ylabel("Percent")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "assembly_line.png", dpi=150)


def main() -> None:
    model = AssemblyLine()
    results = cb.Experiment(
        model, replications=10, window=cb.Window(duration=SIMULATION_TIME),
        seed=RANDOM_SEED,
    ).run()
    if results.failed.any():
        raise RuntimeError(f"{results.failed.sum()} assembly trials failed")
    print_results(model, results)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    (PLOT_DIR / "process_graph.mmd").write_text(mermaid(model) + "\n")


if __name__ == "__main__":
    main()
