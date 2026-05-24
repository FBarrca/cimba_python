![Cimba logo](subprojects/cimba/images/logo_large.jpg)

# Cimba Python

## Fast discrete event simulation for Python

Cimba Python is a Python interface to [Cimba](https://github.com/ambonvik/cimba),
a multithreaded discrete event simulation engine written in C and assembly.

It is designed for Python simulation models that need more speed than pure
Python event scheduling can usually provide. In the included M/M/1 benchmark,
Cimba Python runs about **8.5-8.9x faster than SimPy**, while keeping model code
in Python.

On an AMD Ryzen 7 9700X under WSL Ubuntu 24.04, averaged over 10 runs:

| Benchmark | SimPy | Cimba Python | Cimba C |
| --- | ---: | ---: | ---: |
| Single core, single trial | 3.185 s | 0.375 s | 0.095 s |
| Multicore, 100 trials | 41.820 s | 4.698 s | 1.178 s |

The benchmark data and charts are in
[`benchmarks/AMD_Ryzen_7_9700X_WSL.ods`](benchmarks/AMD_Ryzen_7_9700X_WSL.ods).

## Install

```bash
pip install cimba
```

or with `uv`:

```bash
uv add cimba
```

Python 3.13 or newer is required. The PyPI wheels embed the Cimba C library, so
you do not need to install Cimba separately.

## What is it?

Cimba Python gives Python models access to Cimba's native simulation engine:
processes, event queues, buffers, object queues, priority queues, resources,
conditions, random distributions, time series, and summary statistics.

## What does the code look like?

```python
import cimba


def arrival(queue):
    while True:
        cimba.hold(cimba.exponential(1.0 / 0.75))
        queue.put(1)


def service(queue):
    while True:
        queue.get(1)
        cimba.hold(cimba.exponential(1.0))


with cimba.Simulation(seed=123) as sim:
    queue = cimba.Buffer("Queue")
    queue.start_recording()

    cimba.Process("Arrival", arrival, queue).start()
    cimba.Process("Service", service, queue).start()

    sim.stop_at(1000.0)
    sim.execute()

    queue.stop_recording()
    print(queue.history().summary().mean)
```

More examples, tutorials, topical guides, and the API reference are in the
[documentation](https://fbarrca.github.io/cimba_python/).

## License

Cimba Python is licensed under Apache-2.0. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).

The bundled Cimba C library is also Apache-2.0 licensed. See
`subprojects/cimba/NOTICE` for attribution.
