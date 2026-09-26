"""Tutorial 1.5: a queue station composed into a model."""

import cimba as cb
from cimba import inputs


class MM1Station(cb.Model):
    interarrival: cb.Input[float] = inputs.dist.exponential(mean=1.0 / 0.75)
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


def run_mm1_trial(*, utilization: float, duration: float,
                  warmup: float, seed: int) -> float:
    model = MM1()
    model.station.interarrival = inputs.dist.exponential(mean=1.0 / utilization)
    results = cb.Experiment(
        model, window=cb.Window(warmup=warmup, duration=duration), seed=seed,
    ).run()
    if results.failed.any():
        raise RuntimeError("M/M/1 trial failed")
    return float(results[model].avg_queue_length[0, 0])


def main() -> None:
    avg = run_mm1_trial(utilization=0.75, duration=1.0e6,
                        warmup=1.0e3, seed=46)
    print(f"Avg {avg:.6f}")


if __name__ == "__main__":
    main()
