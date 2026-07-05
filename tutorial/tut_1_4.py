"""Tutorial 1.4: collect queue statistics over a long run."""

import cimba.random as random
import cimba.sim as sim


class MM1(sim.Model):
    utilization: sim.Param
    avg_queue_length: sim.Output
    queue: sim.Queue


model = MM1("MM1")


@model.process
def arrival(env: MM1):
    while True:
        t_ia = random.exponential(1.0 / env.utilization)
        sim.hold(t_ia)
        sim.put(env.queue, 1)


@model.process
def service(env: MM1):
    while True:
        sim.get(env.queue, 1)
        t_srv = random.exponential(1.0)
        sim.hold(t_srv)


@model.collect
def collect_stats(env: MM1):
    env.avg_queue_length = env.queue.history().mean()
    sim.queue_report(env.queue)
    env.queue.history().pacf_correlogram(lags=20)


def main() -> None:
    exp = model.experiment(
        utilization=[0.75],
        replications=1,
        duration=1.0e6,
        warmup=1.0e3,
        seed=45,
    )
    failures = exp.run()
    if failures:
        raise RuntimeError(f"{failures} trial(s) failed")
    avg = float(exp["avg_queue_length"][0])
    print("Theory predicts an average M/M/1 waiting-queue length of 2.25")
    print(f"Simulation result: {avg:.6f}")


if __name__ == "__main__":
    main()
