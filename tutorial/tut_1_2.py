"""Tutorial 1.2: stop the M/M/1 simulation at a fixed duration."""

import cimba
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
    env.avg_queue_length = sim.mean_level(env.queue)


def main() -> None:
    cimba.logger_flags_on(cimba.LOGGER_INFO)
    exp = model.experiment(
        utilization=[0.75],
        replications=1,
        duration=10.0,
        warmup=0.0,
        seed=43,
    )
    failures = exp.run()
    if failures:
        raise RuntimeError(f"{failures} trial(s) failed")
    avg = float(exp["avg_queue_length"][0])
    print(f"Simulation stopped at t=10.0, average queue length: {avg:.6f}")


if __name__ == "__main__":
    main()
