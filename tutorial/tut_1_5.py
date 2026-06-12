"""Tutorial 1.5: refactored M/M/1 trial with parameters and outputs."""

import cimba.sim as sim


class MM1(sim.Model):
    utilization: sim.Param
    avg_queue_length: sim.Output
    queue: sim.Queue


def build_model() -> MM1:
    model = MM1("MM1")

    @model.process
    def arrival(env: MM1):
        while True:
            t_ia = sim.exponential(1.0 / env.utilization)
            sim.hold(t_ia)
            sim.put(env.queue, 1)

    @model.process
    def service(env: MM1):
        while True:
            sim.get(env.queue, 1)
            t_srv = sim.exponential(1.0)
            sim.hold(t_srv)

    @model.collect
    def collect_stats(env: MM1):
        env.avg_queue_length = sim.mean_level(env.queue)

    return model


def run_mm1_trial(
    *,
    utilization: float,
    duration: float,
    warmup: float,
    seed: int,
) -> float:
    exp = build_model().experiment(
        utilization=[utilization],
        replications=1,
        duration=duration,
        warmup=warmup,
        seed=seed,
    )
    failures = exp.run()
    if failures:
        raise RuntimeError(f"{failures} trial(s) failed")
    return float(exp["avg_queue_length"][0])


def main() -> None:
    avg = run_mm1_trial(
        utilization=0.75,
        duration=1.0e6,
        warmup=1.0e3,
        seed=46,
    )
    print(f"Avg {avg:.6f}")


if __name__ == "__main__":
    main()
