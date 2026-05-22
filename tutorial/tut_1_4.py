from dataclasses import dataclass

import cimba


@dataclass
class MM1Trial:
    arr_rate: float = 0.75
    srv_rate: float = 1.0
    warmup_time: float = 50.0
    duration: float = 5000.0
    seed: int = 14
    avg_queue_length: float = 0.0
    arrivals: int = 0
    services: int = 0


def theoretical_queue_length(arr_rate: float, srv_rate: float) -> float:
    rho = arr_rate / srv_rate
    return rho * rho / (1.0 - rho)


def arrival(me, ctx: MM1Trial):
    mean = 1.0 / ctx.arr_rate
    while True:
        cimba.hold(cimba.exponential(mean))
        ctx.arrivals += 1
        ctx.queue.put(1)


def service(me, ctx: MM1Trial):
    mean = 1.0 / ctx.srv_rate
    while True:
        ctx.queue.get(1)
        cimba.hold(cimba.exponential(mean))
        ctx.services += 1


def recorder(me, ctx: MM1Trial):
    cimba.hold(ctx.warmup_time)
    ctx.queue.start_recording()
    cimba.hold(ctx.duration)
    ctx.queue.stop_recording()
    ctx.arrival_process.stop()
    ctx.service_process.stop()
    ctx.simulation.clear()


def run(seed: int = 14) -> MM1Trial:
    trial = MM1Trial(seed=seed)
    with cimba.Simulation(seed=trial.seed) as sim:
        trial.simulation = sim
        trial.queue = cimba.Buffer("Queue")
        trial.arrival_process = cimba.Process("Arrival", arrival, trial).start()
        trial.service_process = cimba.Process("Service", service, trial).start()
        cimba.Process("Recorder", recorder, trial).start()
        sim.execute()

        trial.avg_queue_length = trial.queue.history().summary().mean

    del trial.queue
    del trial.arrival_process
    del trial.service_process
    del trial.simulation
    return trial


def main() -> None:
    trial = run()
    expected = theoretical_queue_length(trial.arr_rate, trial.srv_rate)
    print(f"Avg {trial.avg_queue_length:.3f} Expected {expected:.3f}")


if __name__ == "__main__":
    main()
