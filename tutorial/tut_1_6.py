from collections.abc import Iterable
from dataclasses import dataclass

import cimba


@dataclass
class MM1Trial:
    arr_rate: float
    srv_rate: float
    duration: float
    seed: int
    avg_queue_length: float = 0.0
    arrivals: int = 0
    services: int = 0


def arrival(ctx: MM1Trial):
    mean = 1.0 / ctx.arr_rate
    while True:
        cimba.hold(cimba.exponential(mean))
        ctx.arrivals += 1
        ctx.queue.put(1)


def service(ctx: MM1Trial):
    mean = 1.0 / ctx.srv_rate
    while True:
        ctx.queue.get(1)
        cimba.hold(cimba.exponential(mean))
        ctx.services += 1


def recorder(ctx: MM1Trial):
    ctx.queue.start_recording()
    cimba.hold(ctx.duration)
    ctx.queue.stop_recording()
    ctx.arrival_process.stop()
    ctx.service_process.stop()
    ctx.simulation.clear()


def run_mm1_trial(trial: MM1Trial) -> MM1Trial:
    with cimba.Simulation(seed=trial.seed) as sim:
        trial.simulation = sim
        trial.queue = cimba.Buffer("Queue")
        trial.arrival_process = cimba.Process("Arrival", arrival, trial).start()
        trial.service_process = cimba.Process("Service", service, trial).start()
        cimba.Process("Recorder", recorder, trial).start()
        sim.execute()

        trial.avg_queue_length = trial.queue.history().summary().mean

    del trial.queue
    del trial.arrival_process
    del trial.service_process
    del trial.simulation
    return trial


def run_experiment(
    rhos: Iterable[float] = (0.25, 0.50, 0.75),
    replications: int = 2,
    duration: float = 2500.0,
    seed: int = 1600,
    processes: int | None = 2,
) -> list[dict[str, float]]:
    rhos = tuple(rhos)
    grid = [(rho, rep) for rho in rhos for rep in range(replications)]

    def trial_fn(index, trial_seed):
        rho, rep = grid[index]
        trial = run_mm1_trial(
            MM1Trial(
                arr_rate=rho,
                srv_rate=1.0,
                duration=duration,
                seed=trial_seed,
            )
        )
        return {
            "rho": rho,
            "replication": rep,
            "avg_queue_length": trial.avg_queue_length,
        }

    samples = cimba.run_experiment(trial_fn, n=len(grid), seed=seed, processes=processes)
    rows = []
    for rho in rhos:
        summary = cimba.DataSummary()
        for sample in samples:
            if sample["rho"] == rho:
                summary.add(sample["avg_queue_length"])
        rows.append({"rho": rho, "avg_queue_length": summary.mean})
    return rows


def main() -> None:
    for row in run_experiment():
        print(f"{row['rho']:.3f}\t{row['avg_queue_length']:.6f}")


if __name__ == "__main__":
    main()
