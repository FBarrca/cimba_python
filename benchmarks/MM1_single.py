"""Cimba Python binding version of ``subprojects/cimba/benchmark/MM1_single.c``."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cimba

NUM_OBJECTS = 1_000_000
ARRIVAL_RATE = 0.9
SERVICE_RATE = 1.0


@dataclass
class Trial:
    arr_mean: float
    srv_mean: float
    num_objects: int
    seed: int | None = None
    obj_cnt: int = 0
    sum_wait: float = 0.0

    @property
    def avg_wait(self) -> float:
        return self.sum_wait / self.obj_cnt


def arrival_process(ctx: Trial) -> None:
    queue = ctx.queue
    for _ in range(ctx.num_objects):
        cimba.hold(cimba.exponential(ctx.arr_mean))
        queue.put(cimba.time())


def service_process(ctx: Trial) -> None:
    queue = ctx.queue
    while True:
        sig, arrival_time = queue.get()
        assert sig == cimba.SUCCESS
        cimba.hold(cimba.exponential(ctx.srv_mean))
        ctx.sum_wait += cimba.time() - arrival_time
        ctx.obj_cnt += 1


def run_trial(trial: Trial) -> Trial:
    cimba.logger_flags_off(cimba.LOGGER_INFO)
    with cimba.Simulation(seed=trial.seed) as sim:
        trial.queue = cimba.ObjectQueue("Queue")
        trial.arrival = cimba.Process("Arrival", arrival_process, trial).start()
        trial.service = cimba.Process("Service", service_process, trial).start()
        sim.execute()
        trial.service.stop()

    del trial.queue
    del trial.arrival
    del trial.service
    return trial


def load_trial(num_objects: int, arrival_rate: float, service_rate: float, seed: int | None) -> Trial:
    return Trial(
        arr_mean=1.0 / arrival_rate,
        srv_mean=1.0 / service_rate,
        num_objects=num_objects,
        seed=seed,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Cimba Python M/M/1 single-trial benchmark.")
    parser.add_argument("-n", "--num-objects", type=int, default=NUM_OBJECTS)
    parser.add_argument("--arrival-rate", type=float, default=ARRIVAL_RATE)
    parser.add_argument("--service-rate", type=float, default=SERVICE_RATE)
    parser.add_argument("-s", "--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    trial = run_trial(load_trial(args.num_objects, args.arrival_rate, args.service_rate, args.seed))
    expected = 1.0 / (args.service_rate - args.arrival_rate)
    print(f"Average system time {trial.avg_wait:f} (expected {expected:f})")


if __name__ == "__main__":
    main()
