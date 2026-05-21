# cimba (Python)

Python bindings for [**Cimba**](https://github.com/ambonvik/cimba) — a
multithreaded discrete-event-simulation library written in C (POSIX pthreads for
parallel replications, stackful coroutines for concurrent processes per thread).

> **Status: early bindings.** The package wraps the core single-thread Cimba
> simulation API: `Simulation`, `Process`, buffers, object/priority queues,
> resources, conditions, random distributions, and summary/time-series helpers.
> Parallel experiment orchestration is still a later layer.

## How it's put together

| Piece | Where | Why |
| --- | --- | --- |
| Native C library | `subprojects/cimba/` (git submodule) | Lives where Meson expects subprojects; pinned to a commit; updated via git |
| Build backend | `meson-python` (`pyproject.toml`) | Upstream already uses Meson; reuses its NASM/C23/pthreads build for free |
| Build wiring | `meson.build` | Pulls the subproject (`my_lib_dep`), builds it **static**, links the extension against it |
| Python package | `src/cimba/` | `src`-layout; `_cimba.pyx` is the Cython extension |
| Tests | `tests/` | Smoke test that the extension imports and calls into C |

The C library is built as a **static** archive and embedded into the
`_cimba` extension, so a built wheel is self-contained (one `.so`, nothing to
ship alongside it).

## Prerequisites

`uv` provides Python 3.13, `meson`, `ninja`, `cython`, and `meson-python`. You
only need these on the system itself:

| Tool | Why | Check |
| --- | --- | --- |
| **uv** | drives the build/test/run workflow | `uv --version` |
| **git** | the C library is a submodule | `git --version` |
| **C compiler** (gcc/clang) | compiles Cimba + the extension | `cc --version` |
| **NASM** | Cimba's ziggurat RNG is in assembly | `nasm --version` |

Cimba also links `pthreads` and `libm`, which are part of the C runtime. On
Ubuntu/WSL the compiler + NASM come from: `sudo apt install build-essential nasm`.

## Quick start (from a fresh clone)

```bash
git clone <repo-url> cimba_python
cd cimba_python
git submodule update --init --recursive    # pulls subprojects/cimba

uv sync                       # creates .venv, installs deps, compiles cimba (~15s first run)
uv run python -c "import cimba; print(cimba.native_version())"   # -> 3.0.0-beta
uv run pytest                 # runs the wrapper smoke/behavior tests
```

No manual `.venv` activation needed — `uv run` handles it. The first `uv sync`
compiles the C library + Cython extension; later runs are incremental.

## Minimal simulation

```python
import cimba


def arrival(me, queue):
    while True:
        cimba.hold(cimba.exponential(1.0 / 0.75))
        queue.put(1)


def service(me, queue):
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

`Simulation` owns Cimba's thread-local event queue and random generator. Objects
created while a simulation is active are kept alive by the simulation and closed
in reverse creation order, so `Process(...).start()` is safe even if you do not
store the process in a local variable.

## Build a wheel

```bash
uv build --wheel              # -> dist/cimba-<ver>-cp313-cp313-<platform>.whl
# or `uv build` for wheel + sdist
```

The wheel statically embeds Cimba, so it needs no system Cimba at runtime. Verify
that in a throwaway environment (no project, no build tools):

```bash
uv run --no-project --isolated \
  --with dist/cimba-*.whl \
  python -c "import cimba; print(cimba.native_version())"   # -> 3.0.0-beta
```

## Troubleshooting

**`FileNotFoundError: .../build/cp313` on import.** `build/cp313/` is the
editable dev install's build directory. If it's deleted, `uv sync` won't
recompile (it reinstalls cimba from uv's cache), so the import breaks. Force a
real rebuild:

```bash
uv sync --reinstall-package cimba
```

Use the same command if a plain `uv sync` doesn't pick up changes to
`meson.build` or `pyproject.toml`. (A genuine fresh clone with a cold uv cache
builds correctly with plain `uv sync`; `uv build` wheels are unaffected.)

## Updating the bundled C library

```bash
cd subprojects/cimba
git fetch && git checkout <commit-or-tag>
cd ../..
git add subprojects/cimba && git commit -m "Bump Cimba to <ref>"
```

## License

This wrapper is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)). It bundles
the Apache-2.0 licensed Cimba library in compiled form; see [`NOTICE`](NOTICE)
and `subprojects/cimba/NOTICE` for attribution.
