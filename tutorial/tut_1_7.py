import argparse
from dataclasses import dataclass

import cimba


@dataclass
class MM1Trial:
    arr_rate: float = 0.75
    srv_rate: float = 1.0
    warmup_time: float = 1000.0
    duration: float = 1.0e6
    seed: int = 17
    avg_queue_length: float = 0.0
    arrivals: int = 0
    services: int = 0


def arrival(me, ctx: MM1Trial):
    mean = 1.0 / ctx.arr_rate
    while True:
        cimba.hold(cimba.exponential(mean))
        ctx.arrivals += 1
        ctx.queue.put(1)


def service(me, ctx: MM1Trial):
    mean = 1.0 / ctx.srv_rate
    while True:
        ctx.queue.get(1)
        cimba.hold(cimba.exponential(mean))
        ctx.services += 1


def recorder(me, ctx: MM1Trial):
    if ctx.warmup_time > 0.0:
        cimba.hold(ctx.warmup_time)
    ctx.queue.start_recording()
    cimba.hold(ctx.duration)
    ctx.queue.stop_recording()
    ctx.arrival_process.stop()
    ctx.service_process.stop()
    ctx.simulation.clear()


def run(trial: MM1Trial) -> MM1Trial:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Python Cimba M/M/1 tutorial trial.")
    parser.add_argument("--arr-rate", type=float, default=0.75)
    parser.add_argument("--srv-rate", type=float, default=1.0)
    parser.add_argument("--warmup-time", type=float, default=1000.0)
    parser.add_argument("--duration", type=float, default=1.0e6)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def run_from_args(argv: list[str] | None = None) -> MM1Trial:
    args = parse_args(argv)
    return run(
        MM1Trial(
            arr_rate=args.arr_rate,
            srv_rate=args.srv_rate,
            warmup_time=args.warmup_time,
            duration=args.duration,
            seed=args.seed,
        )
    )


def main(argv: list[str] | None = None) -> None:
    trial = run_from_args(argv)
    print(f"Avg {trial.avg_queue_length:.6f}")


if __name__ == "__main__":
    main()
