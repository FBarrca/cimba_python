"""
Cimba version of the SimPy/salabim assembly-line comparison model.

Run from the repository root:

    uv run --extra plot python tutorial/assembly_line.py
"""

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script needs matplotlib. Run it with: "
        "uv run --extra plot python tutorial/assembly_line.py"
    ) from exc

import cimba as cp
import cimba.random as random
import cimba.sim as sim


RANDOM_SEED = 45
STATION_1_NAME = "Station 1"
STATION_2_NAME = "Station 2"
STATION_3_NAME = "Station 3"
STATION_NAMES = (STATION_1_NAME, STATION_2_NAME, STATION_3_NAME)
NUM_STATIONS = 3
STATION_1_MEAN = 5.0
STATION_2_MEAN = 7.0
STATION_3_MEAN = 4.0
INTERARRIVAL_TIME = 3.0
SIMULATION_TIME = 10_000.0
PLOT_DIR = Path(__file__).with_name("assembly_line_plots")


class Part(sim.Struct):
    part_id: int
    arrival_system: float
    station_entry: float


class Station(sim.Component):
    avg_wait_time: sim.Output
    utilization: sim.Output
    inbox: sim.Store
    downstream: sim.Ref[sim.Component]
    resource: sim.Resource
    wait_time: sim.Dataset

    def __init__(self, name: str, mean_processing_time: float, *,
                 downstream=None):
        self.name = name
        self.mean_processing_time = mean_processing_time
        if downstream is not None:
            self.downstream = downstream

    @sim.collect
    def station_stats(self, env):
        self.avg_wait_time = self.wait_time.mean()
        self.utilization = 100.0 * sim.mean_in_use(self.resource)

    @sim.process
    def server(self, env):
        while True:
            # Take a part from the inbox.
            handle = sim.store_take(self.inbox)
            item = Part(handle)

            # Update the part's wait time.
            wait_time = sim.now() - item.station_entry
            self.wait_time.add(wait_time)

            # Hold the resource for the processing time.
            sim.acquire(self.resource)
            sim.hold(random.exponential(self.mean_processing_time))
            sim.release(self.resource)

            # Update the part's station entry time to the current time.
            item.station_entry = sim.now()

            # Put the part in the downstream station's inbox.
            sim.store_put(self.downstream.inbox, handle)


class FinishedParts(sim.Component):
    inbox: sim.Store
    departed: sim.Store

    @sim.process
    def finish(self, env):
        while True:
            handle = sim.store_take(self.inbox)
            item = Part(handle)
            env.cycle_time.add(sim.now() - item.arrival_system)
            sim.get(env.system, 1)
            sim.store_put(self.departed, handle)

    @sim.process
    def reclaim(self, env):
        while True:
            sim.despawn(sim.store_take(self.departed))


class AssemblyLine(sim.Model):
    total_parts_produced: sim.Output
    avg_cycle_time: sim.Output
    max_cycle_time: sim.Output
    throughput_rate: sim.Output
    avg_number_in_system: sim.Output
    max_number_in_system: sim.Output
    final_number_in_system: sim.Output

    generated_parts: sim.State
    part_lifecycle: sim.Spawnable
    system: sim.Queue
    cycle_time: sim.Dataset
    finished_parts: FinishedParts = FinishedParts()
    station_3: Station = Station(STATION_3_NAME, STATION_3_MEAN,
                                 downstream=finished_parts)
    station_2: Station = Station(STATION_2_NAME, STATION_2_MEAN,
                                 downstream=station_3)
    station_1: Station = Station(STATION_1_NAME, STATION_1_MEAN,
                                 downstream=station_2)


def build_model(raw_dir: Path) -> AssemblyLine:
    cycle_file = sim.log_text(str(raw_dir / "cycle_times.txt"))
    wait_files = (
        sim.log_text(str(raw_dir / "station_1_wait_times.txt")),
        sim.log_text(str(raw_dir / "station_2_wait_times.txt")),
        sim.log_text(str(raw_dir / "station_3_wait_times.txt")),
    )
    system_file = sim.log_text(str(raw_dir / "number_in_system.txt"))

    model = AssemblyLine("assembly_line")

    @model.process
    def arrivals(env: AssemblyLine):
        while True:
            sim.hold(random.exponential(INTERARRIVAL_TIME))
            handle = sim.spawn(env.part_lifecycle, env)
            part = Part(handle)

            env.generated_parts += 1
            part.part_id = env.generated_parts
            part.arrival_system = sim.now()

    @model.process
    def part_lifecycle(env: AssemblyLine, item: Part):
        sim.put(env.system, 1)
        item.station_entry = sim.now()
        sim.store_put(env.station_1.inbox, sim.current())

    @model.collect
    def collect_stats(env: AssemblyLine):
        completed = env.cycle_time.count()
        env.total_parts_produced = completed
        env.avg_cycle_time = env.cycle_time.mean()
        env.max_cycle_time = env.cycle_time.max()
        env.throughput_rate = completed / env.duration_s
        env.avg_number_in_system = sim.mean_level(env.system)
        env.max_number_in_system = sim.timeseries_max(
            sim.queue_history(env.system)
        )
        env.final_number_in_system = sim.level(env.system)

        env.cycle_time.print_file(cycle_file, 0)
        env.station_1.wait_time.print_file(wait_files[0], 0)
        env.station_2.wait_time.print_file(wait_files[1], 0)
        env.station_3.wait_time.print_file(wait_files[2], 0)
        sim.timeseries_print_file(sim.queue_history(env.system), system_file, 0)

    return model


def read_values(path: Path) -> np.ndarray:
    text = path.read_text().strip()
    if not text:
        return np.array([], dtype=float)
    return np.atleast_1d(np.loadtxt(path, dtype=float))


def read_timeseries(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text().strip()
    if not text:
        empty = np.array([], dtype=float)
        return empty, empty
    rows = np.atleast_2d(np.loadtxt(path, dtype=float))
    return rows[:, 0], rows[:, 1]


def station_values(exp: sim.Experiment, field: str) -> np.ndarray:
    return np.array(
        [
            exp[f"station_1__{field}"][0],
            exp[f"station_2__{field}"][0],
            exp[f"station_3__{field}"][0],
        ],
        dtype=float,
    )


def print_results(exp: sim.Experiment) -> None:
    wait_times = station_values(exp, "avg_wait_time")
    utilization = station_values(exp, "utilization")

    print("--- Simulation Finished ---")
    print("\n--- Simulation Results Analysis (Cimba) ---")
    print(f"Total parts produced: {int(exp['total_parts_produced'][0])}")
    print(
        "Average cycle time per part: "
        f"{exp['avg_cycle_time'][0]:.2f} minutes"
    )
    print(
        "Maximum cycle time per part: "
        f"{exp['max_cycle_time'][0]:.2f} minutes"
    )
    print(f"Throughput rate: {exp['throughput_rate'][0]:.2f} parts per minute")
    for i in range(NUM_STATIONS):
        print(
            f"{STATION_NAMES[i]} - Average Wait Time: "
            f"{wait_times[i]:.2f} minutes"
        )
        print(f"{STATION_NAMES[i]} - Utilization: {utilization[i]:.2f}%")
    print(f"Average number in system: {exp['avg_number_in_system'][0]:.2f}")
    print(f"Maximum number in system: {exp['max_number_in_system'][0]:.0f}")
    print(f"Parts still in system: {exp['final_number_in_system'][0]:.0f}")


def plot_process_dag(model: AssemblyLine) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    graph = model.process_dag()
    mermaid_path = PLOT_DIR / "process_dag.mmd"
    dot_path = PLOT_DIR / "process_dag.dot"

    mermaid_path.write_text(graph.to_mermaid(direction="TD") + "\n")
    dot_path.write_text(graph.to_dot(rankdir="TB") + "\n")

    dot = shutil.which("dot")
    if dot is not None:
        subprocess.run(
            [dot, "-Tpng", str(dot_path), "-o",
             str(PLOT_DIR / "process_dag.png")],
            check=True,
        )
        subprocess.run(
            [dot, "-Tsvg", str(dot_path), "-o",
             str(PLOT_DIR / "process_dag.svg")],
            check=True,
        )

    print(f"\nSaved process DAG in {PLOT_DIR}")


def plot_results(exp: sim.Experiment, raw_dir: Path) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    cycle_times = read_values(raw_dir / "cycle_times.txt")
    wait_times = [
        read_values(raw_dir / f"station_{i + 1}_wait_times.txt")
        for i in range(NUM_STATIONS)
    ]
    time_points, parts_in_system = read_timeseries(
        raw_dir / "number_in_system.txt"
    )

    plt.figure(figsize=(10, 6))
    plt.hist(cycle_times, bins=20, color="skyblue", edgecolor="black")
    avg_cycle_time = exp["avg_cycle_time"][0]
    plt.axvline(
        avg_cycle_time,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Avg: {avg_cycle_time:.2f}",
    )
    plt.title("Distribution of Part Cycle Times")
    plt.xlabel("Cycle Time (minutes)")
    plt.ylabel("Number of Parts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cycle_times.png", dpi=150)

    fig, axes = plt.subplots(
        NUM_STATIONS, 1, figsize=(10, 4 * NUM_STATIONS), sharex=True
    )
    fig.suptitle("Distribution of Waiting Times at Each Station", fontsize=16)
    avg_wait_times = station_values(exp, "avg_wait_time")
    for i, ax in enumerate(axes):
        ax.hist(wait_times[i], bins=15, color="lightcoral", edgecolor="black")
        ax.axvline(
            avg_wait_times[i],
            color="blue",
            linestyle="dashed",
            linewidth=2,
            label=f"Avg: {avg_wait_times[i]:.2f}",
        )
        ax.set_title(f"{STATION_NAMES[i]} Waiting Times")
        ax.set_ylabel("Number of Parts")
        ax.legend()
    axes[-1].set_xlabel("Waiting Time (minutes)")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(PLOT_DIR / "station_wait_times.png", dpi=150)

    station_utilization = station_values(exp, "utilization")
    plt.figure(figsize=(10, 6))
    plt.bar(STATION_NAMES, station_utilization, color="mediumseagreen")
    plt.title("Average Station Utilization")
    plt.xlabel("Station")
    plt.ylabel("Utilization (%)")
    plt.ylim(0, 100)
    for i, value in enumerate(station_utilization):
        plt.text(i, value + 1, f"{value:.2f}%", ha="center")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "station_utilization.png", dpi=150)

    plt.figure(figsize=(12, 6))
    plt.step(time_points, parts_in_system, where="post", color="dodgerblue")
    avg_number = exp["avg_number_in_system"][0]
    plt.axhline(
        avg_number,
        color="blue",
        linewidth=1,
        label=f"Mean: {avg_number:.2f}",
    )
    plt.title("Number of Parts in the System Over Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Number of Parts")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "number_in_system.png", dpi=150)

    print(f"\nSaved plots in {PLOT_DIR}")
    if "agg" not in plt.get_backend().lower():
        plt.show()


def main() -> None:
    print("--- Assembly Line Simulation Starting (Cimba) ---")
    print(f"cimba {cp.version()}")

    with TemporaryDirectory() as temp_dir:
        raw_dir = Path(temp_dir)
        model = build_model(raw_dir)
        exp = model.experiment(
            replications=1000,
            duration=SIMULATION_TIME,
            warmup=0.0,
            seed=RANDOM_SEED,
        )
        failures = exp.run()
        if failures:
            raise RuntimeError(f"{failures} trial(s) failed")

        print_results(exp)
        plot_process_dag(model)
        # plot_results(exp, raw_dir)

    print("\n--- End of Cimba Script ---")


if __name__ == "__main__":
    main()
