from dataclasses import dataclass

import cimba


@dataclass
class MM1Trial:
    arr_rate: float = 0.75
    srv_rate: float = 1.0
    duration: float = 1000.0
    seed: int = 123
    avg_queue_length: float = 0.0
    arrivals: int = 0
    services: int = 0


def arrival(ctx: MM1Trial):
    mean = 1.0 / ctx.arr_rate
    while True:
        cimba.hold(cimba.exponential(mean))
        ctx.arrivals += 1
        ctx.queue.put(1)


def service(ctx: MM1Trial):
    mean = 1.0 / ctx.srv_rate
    while True:
        ctx.queue.get(1)
        cimba.hold(cimba.exponential(mean))
        ctx.services += 1


def end_sim(subject, ctx: MM1Trial):
    ctx.arrival_process.stop()
    ctx.service_process.stop()
    ctx.queue.stop_recording()
    ctx.simulation.clear()


def run(seed: int = 11, stop_time: float = 25.0) -> MM1Trial:
    trial = MM1Trial(duration=stop_time, seed=seed)
    with cimba.Simulation(seed=trial.seed) as sim:
        trial.simulation = sim
        trial.queue = cimba.Buffer("Queue")
        trial.queue.start_recording()
        trial.arrival_process = cimba.Process("Arrival", arrival, trial).start()
        trial.service_process = cimba.Process("Service", service, trial).start()
        sim.schedule(end_sim, trial.duration, obj=trial)
        sim.execute()

        trial.avg_queue_length = trial.queue.history().summary().mean

    del trial.queue
    del trial.arrival_process
    del trial.service_process
    del trial.simulation
    return trial


def main() -> None:
    trial = run()
    print(f"Arrivals {trial.arrivals} Services {trial.services} Avg {trial.avg_queue_length:.3f}")


if __name__ == "__main__":
    main()
