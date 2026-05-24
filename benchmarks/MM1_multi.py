"""Cimba Python binding version of ``subprojects/cimba/benchmark/MM1_multi.c``."""

from __future__ import annotations

import argparse
import math
import multiprocessing
import statistics
from dataclasses import dataclass

import cimba

NUM_OBJECTS = 1_000_000
ARRIVAL_RATE = 0.9
SERVICE_RATE = 1.0
NUM_TRIALS = 100
MASK64 = (1 << 64) - 1


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


def fmix64(seed: int, nonce: int) -> int:
    h = (seed + nonce) & MASK64
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & MASK64
    h ^= h >> 33
    h = (h * 0xC4CEB9FE1A85EC53) & MASK64
    h ^= h >> 33
    return h & MASK64


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


def run_trial_from_args(args: tuple[int, float, float, int]) -> float:
    num_objects, arrival_rate, service_rate, seed = args
    trial = run_trial(
        Trial(
            arr_mean=1.0 / arrival_rate,
            srv_mean=1.0 / service_rate,
            num_objects=num_objects,
            seed=seed,
        )
    )
    return trial.avg_wait


def run_experiment(
    *,
    num_objects: int,
    arrival_rate: float,
    service_rate: float,
    num_trials: int,
    seed: int,
    processes: int | None,
) -> list[float]:
    trial_args = [
        (num_objects, arrival_rate, service_rate, fmix64(seed, idx))
        for idx in range(num_trials)
    ]
    with multiprocessing.Pool(processes=processes) as pool:
        return pool.map(run_trial_from_args, trial_args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Cimba Python M/M/1 multi-trial benchmark.")
    parser.add_argument("-n", "--num-objects", type=int, default=NUM_OBJECTS)
    parser.add_argument("-r", "--num-trials", type=int, default=NUM_TRIALS)
    parser.add_argument("-j", "--processes", type=int, default=None)
    parser.add_argument("--arrival-rate", type=float, default=ARRIVAL_RATE)
    parser.add_argument("--service-rate", type=float, default=SERVICE_RATE)
    parser.add_argument("-s", "--seed", type=int, default=123)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = run_experiment(
        num_objects=args.num_objects,
        arrival_rate=args.arrival_rate,
        service_rate=args.service_rate,
        num_trials=args.num_trials,
        seed=args.seed,
        processes=args.processes,
    )
    n = len(results)
    if n > 1:
        mean_tsys = statistics.mean(results)
        sdev_tsys = statistics.stdev(results)
        serr_tsys = sdev_tsys / math.sqrt(n)
        ci_w = 1.96 * serr_tsys
        ci_l = mean_tsys - ci_w
        ci_u = mean_tsys + ci_w
        expected = 1.0 / (args.service_rate - args.arrival_rate)
        print(f"Average system time {mean_tsys:f} (n {n}, conf.int. {ci_l:f} - {ci_u:f}, expected {expected:f})")


if __name__ == "__main__":
    main()
