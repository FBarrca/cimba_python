"""Tutorial 1.6: parallel M/M/1 utilization sweep."""

import numpy as np

import cimba as cb
from cimba import inputs


class MM1Station(cb.Model):
    interarrival: cb.Input[float] = inputs.dist.exponential(mean=1.0)
    service_time: cb.Input[float] = inputs.dist.exponential(mean=1.0)
    queue: cb.Container

    @cb.process
    def arrival(self):
        while True:
            cb.hold(self.interarrival.next())
            self.queue.put(1)

    @cb.process
    def service(self):
        while True:
            self.queue.get(1)
            cb.hold(self.service_time.next())


class MM1(cb.Model):
    station: MM1Station
    avg_queue_length: cb.Output[float]

    def __init__(self):
        self.station = MM1Station()

    @cb.on_end
    def collect_stats(self):
        self.avg_queue_length = self.station.queue.mean_level()


def sweep_rho(*, replications: int = 10, duration: float = 1.0e6,
              warmup: float = 1.0e3, seed: int = 42
              ) -> tuple[np.ndarray, np.ndarray]:
    rhos = np.arange(0.025, 1.0, 0.025)
    model = MM1()
    axis = cb.sweep(*rhos)
    model.station.interarrival = inputs.dist.exponential(
        mean=axis.map(lambda rho: 1.0 / rho))
    results = cb.Experiment(
        model, replications=replications,
        window=cb.Window(warmup=warmup, duration=duration), seed=seed,
    ).run()
    if results.failed.any():
        raise RuntimeError("M/M/1 sweep failed")
    return rhos, results[model].avg_queue_length.values


def print_sweep(rhos: np.ndarray, values: np.ndarray) -> None:
    print(f"cimba {cb.engine_version()}")
    print(f"{'rho':>8} {'simulated':>10} {'+/-95%':>10} {'theory':>10}")
    for rho, samples in zip(rhos, values):
        mean = float(samples.mean())
        ci = 0.0 if samples.size < 2 else float(
            1.96 * samples.std(ddof=1) / np.sqrt(samples.size))
        theory = rho * rho / (1.0 - rho)
        print(f"{rho:8.3f} {mean:10.4f} {ci:10.4f} {theory:10.4f}")


def main() -> None:
    rhos, values = sweep_rho()
    print_sweep(rhos, values)


if __name__ == "__main__":
    main()
