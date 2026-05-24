from dataclasses import dataclass

import cimba

USERFLAG1 = 0x00000001


@dataclass
class MM1Trial:
    arr_rate: float = 0.75
    srv_rate: float = 1.0
    duration: float = 20.0
    seed: int = 13
    avg_queue_length: float = 0.0
    arrivals: int = 0
    services: int = 0
    trace: list[str] | None = None


def log_user(ctx: MM1Trial, message: str, *args: object) -> None:
    if ctx.trace is not None:
        ctx.trace.append(message % args)


def arrival(ctx: MM1Trial):
    mean = 1.0 / ctx.arr_rate
    while True:
        t_ia = cimba.exponential(mean)
        log_user(ctx, "Holds for %f time units", t_ia)
        cimba.hold(t_ia)
        ctx.arrivals += 1
        log_user(ctx, "Puts one into the queue")
        ctx.queue.put(1)


def service(ctx: MM1Trial):
    mean = 1.0 / ctx.srv_rate
    while True:
        log_user(ctx, "Gets one from the queue")
        ctx.queue.get(1)
        t_srv = cimba.exponential(mean)
        log_user(ctx, "Got one, services it for %f time units", t_srv)
        cimba.hold(t_srv)
        ctx.services += 1


def end_sim(ctx: MM1Trial):
    cimba.hold(ctx.duration)
    log_user(ctx, "--- Game Over ---")
    ctx.arrival_process.stop()
    ctx.service_process.stop()
    ctx.queue.stop_recording()
    ctx.simulation.clear()


def run(seed: int = 13) -> MM1Trial:
    cimba.logger_flags_off(cimba.LOGGER_INFO)
    cimba.logger_flags_off(USERFLAG1)
    trial = MM1Trial(seed=seed, trace=[])
    with cimba.Simulation(seed=trial.seed) as sim:
        trial.simulation = sim
        trial.queue = cimba.Buffer("Queue")
        trial.queue.start_recording()
        trial.arrival_process = cimba.Process("Arrival", arrival, trial).start()
        trial.service_process = cimba.Process("Service", service, trial).start()
        cimba.Process("EndSimulation", end_sim, trial).start()
        sim.execute()

        trial.avg_queue_length = trial.queue.history().summary().mean

    del trial.queue
    del trial.arrival_process
    del trial.service_process
    del trial.simulation
    return trial


def main() -> None:
    trial = run()
    print(f"Avg {trial.avg_queue_length:.3f}")


if __name__ == "__main__":
    main()
