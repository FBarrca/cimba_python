"""Tutorial 1.3: user logging from Python process bodies."""

import cimba
import cimba.random as random
import cimba.sim as sim

USERFLAG1 = 0x00000001

MSG_ARR_HOLD = sim.log_text("Holds for")
MSG_ARR_PUT = sim.log_text("Puts one into the queue")
MSG_SRV_GET = sim.log_text("Gets one from the queue")
MSG_SRV_HOLD = sim.log_text("Got one, services it for")


class MM1(sim.Model):
    utilization: sim.Param
    avg_queue_length: sim.Output
    queue: sim.Queue


model = MM1("MM1")


@model.process
def arrival(env: MM1):
    while True:
        t_ia = random.exponential(1.0 / env.utilization)
        sim.log_user_f64(USERFLAG1, MSG_ARR_HOLD, t_ia)
        sim.hold(t_ia)
        sim.log_user(USERFLAG1, MSG_ARR_PUT)
        sim.put(env.queue, 1)


@model.process
def service(env: MM1):
    while True:
        sim.log_user(USERFLAG1, MSG_SRV_GET)
        sim.get(env.queue, 1)
        t_srv = random.exponential(1.0)
        sim.log_user_f64(USERFLAG1, MSG_SRV_HOLD, t_srv)
        sim.hold(t_srv)


@model.collect
def collect_stats(env: MM1):
    env.avg_queue_length = sim.mean_level(env.queue)


def main() -> None:
    cimba.logger_flags_off(cimba.LOGGER_INFO)
    cimba.logger_flags_on(USERFLAG1)
    exp = model.experiment(
        utilization=[0.75],
        replications=1,
        duration=10.0,
        warmup=0.0,
        seed=44,
    )
    failures = exp.run()
    if failures:
        raise RuntimeError(f"{failures} trial(s) failed")
    avg = float(exp["avg_queue_length"][0])
    print(f"Average queue length with user logging enabled: {avg:.6f}")
    cimba.logger_flags_off(USERFLAG1)


if __name__ == "__main__":
    main()
