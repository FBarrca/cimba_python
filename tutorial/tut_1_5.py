from dataclasses import dataclass

import cimba


@dataclass
class MM1Trial:
    arr_rate: float = 0.75
    srv_rate: float = 1.0
    warmup_time: float = 1000.0
    duration: float = 1.0e6
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


def recorder(ctx: MM1Trial):
    if ctx.warmup_time > 0.0:
        cimba.hold(ctx.warmup_time)
    ctx.queue.start_recording()
    cimba.hold(ctx.duration)
    ctx.queue.stop_recording()
    ctx.arrival_process.stop()
    ctx.service_process.stop()
    ctx.simulation.clear()


def run_mm1_trial(trial: MM1Trial) -> MM1Trial:
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


def load_params() -> MM1Trial:
    return MM1Trial(arr_rate=0.75, srv_rate=1.0, warmup_time=1000.0, duration=1.0e6)


def run(trial: MM1Trial | None = None) -> MM1Trial:
    return run_mm1_trial(trial if trial is not None else load_params())


def main() -> None:
    trial = run(MM1Trial(warmup_time=20.0, duration=1500.0, seed=15))
    print(f"Avg {trial.avg_queue_length:.6f}")


if __name__ == "__main__":
    main()
