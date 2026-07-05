"""Tutorial 1.6: parallel M/M/1 utilization sweep."""

import numpy as np

import cimba
import cimba.random as random
import cimba.sim as sim


class MM1Station(sim.Component):
    queue: sim.Queue

    @sim.process
    def arrival(self, env):
        while True:
            t_ia = random.exponential(1.0 / env.utilization)
            sim.hold(t_ia)
            sim.put(self.queue, 1)

    @sim.process
    def service(self, env):
        while True:
            sim.get(self.queue, 1)
            t_srv = random.exponential(1.0)
            sim.hold(t_srv)


class MM1(sim.Model):
    utilization: sim.Param
    avg_queue_length: sim.Output
    station: MM1Station = MM1Station()


def build_model() -> MM1:
    model = MM1("MM1")

    @model.collect
    def collect_stats(env: MM1):
        env.avg_queue_length = sim.mean_level(env.station.queue)

    return model


def sweep_rho(
    *,
    replications: int = 10,
    duration: float = 1.0e6,
    warmup: float = 1.0e3,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rhos = np.arange(0.025, 1.0, 0.025)
    exp = build_model().experiment(
        utilization=rhos,
        replications=replications,
        duration=duration,
        warmup=warmup,
        seed=seed,
    )
    failures = exp.run()
    if failures:
        raise RuntimeError(f"{failures} trial(s) failed")
    return rhos, exp["avg_queue_length"].reshape(len(rhos), replications)


def print_sweep(rhos: np.ndarray, values: np.ndarray) -> None:
    print(f"cimba {cimba.native_version()}")
    print(f"{'rho':>8} {'simulated':>10} {'+/-95%':>10} {'theory':>10}")
    for rho, samples in zip(rhos, values):
        mean = float(samples.mean())
        ci = 0.0
        if samples.size > 1:
            ci = float(1.96 * samples.std(ddof=1) / np.sqrt(samples.size))
        theory = rho * rho / (1.0 - rho)
        print(f"{rho:8.3f} {mean:10.4f} {ci:10.4f} {theory:10.4f}")


def main() -> None:
    rhos, values = sweep_rho(replications=10, duration=1.0e6, warmup=1.0e3)
    print_sweep(rhos, values)


if __name__ == "__main__":
    main()
