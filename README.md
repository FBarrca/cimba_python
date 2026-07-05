![Cimba logo](docs/static/cimba_logo_large.jpg)

# Cimba Python

## Fast discrete event simulation for Python

Cimba Python is a Python interface to [Cimba](https://github.com/ambonvik/cimba),
a multithreaded discrete event simulation engine written in C and assembly.

It is designed for Python simulation models that need more speed than pure
Python event scheduling can usually provide. In the included M/M/1 benchmark,
Cimba Python runs about **27-33x faster than SimPy** after its one-time Numba
compile, while keeping model code in Python.

On an AMD Ryzen 7 9700X under WSL Ubuntu 24.04, averaged over 10 runs:

| Benchmark | SimPy | Cimba Python | Cimba C |
| --- | ---: | ---: | ---: |
| Single core, single trial | 2.612 s | 0.096 s | 0.083 s |
| Multicore, 100 trials | 36.807 s | 1.131 s | 0.970 s |

The benchmark data and charts are in
[`benchmark/AMD_Ryzen_7_9700X_WSL.ods`](benchmark/AMD_Ryzen_7_9700X_WSL.ods).

## Install

```bash
pip install cimba
```

or with `uv`:

```bash
uv add cimba
```

Python 3.13 or newer is required. The wheel embeds the Cimba C library, so you
do not need to install Cimba separately.

## What is it?

Cimba Python gives Python models access to Cimba's native simulation engine
through the `cimba.sim` API: processes, event queues, buffers, queues, stores,
priority queues, resources, resource pools, conditions, timers, events, logging
helpers, and experiment tables. Random distributions live in `cimba.random`.

## What does the code look like?

```python
import cimba.sim as sim
import cimba.random as random


class MM1(sim.Model):
    utilization: sim.Param
    avg_queue_length: sim.Output
    queue: sim.Queue


model = MM1("MM1")


@model.process
def arrival(env: MM1):
    while True:
        sim.hold(random.exponential(1.0 / env.utilization))
        env.queue.put(1)


@model.process
def service(env: MM1):
    while True:
        env.queue.get(1)
        sim.hold(random.exponential(1.0))


@model.collect
def collect_stats(env: MM1):
    env.avg_queue_length = env.queue.mean_level()


exp = model.experiment(
    utilization=0.75,
    replications=100,
    duration=1000.0,
    warmup=100.0,
    seed=123,
)
exp.run()

print(exp["avg_queue_length"].mean())
```

More examples, tutorials, background notes, and the API reference are in the
[documentation](https://fbarrca.github.io/cimba_python/).

## Development

From a fresh clone:

```bash
git submodule update --init --recursive
uv sync
uv run pytest
```

## License

Cimba Python is licensed under Apache-2.0. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).

The bundled Cimba C library is also Apache-2.0 licensed. See
`subprojects/cimba/NOTICE` for attribution.
