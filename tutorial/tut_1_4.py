"""Tutorial 1.4: long-run queue statistics after a warmup."""

import cimba as cb
from cimba import inputs


class MM1(cb.Model):
    interarrival: cb.Input[float] = inputs.dist.exponential(mean=1.0 / 0.75)
    service_time: cb.Input[float] = inputs.dist.exponential(mean=1.0)
    queue: cb.Container
    avg_queue_length: cb.Output[float]

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

    @cb.on_end
    def collect_stats(self):
        self.avg_queue_length = self.queue.mean_level()


def main() -> None:
    model = MM1()
    results = cb.Experiment(
        model, window=cb.Window(warmup=100.0, duration=5_000.0), seed=14,
    ).run()
    if results.failed.any():
        raise RuntimeError("M/M/1 trial failed")
    print(f"Average queue length: {results[model].avg_queue_length[0, 0]:.6f}")


if __name__ == "__main__":
    main()
