# Python bindings for Cimba, a multithreaded discrete-event-simulation library.

This package wraps the native Cimba library (vendored under `subprojects/cimba`)
with two modules:

| Module | Role |
| --- | --- |
| `cimba.sim` | SimPy-flavored `Model` / `@process` API compiled via Numba |

`cimba.native_version()` (alias `cimba.version()`) calls into the linked C
library so you can confirm the toolchain works end to end.

## How it's put together

| Piece | Where | Why |
| --- | --- | --- |
| Native C library | `subprojects/cimba/` (git submodule) | Meson subproject; pinned commit |
| Build backend | `meson-python` (`pyproject.toml`) | Reuses upstream Meson/NASM/C23 build |
| Build wiring | `meson.build` | cffi glue + native shims, `link_whole` libcimba |
| Python package | `src/cimba/` | `sim` module + `_cimba` cffi extension |
| Examples | `examples/` | M/G/1 demos and benchmarks |
| Tests | `tests/` | Import, linkage, and a short sim-API smoke run |

The C library is built as a **static** archive and embedded into `_cimba`, so a
wheel is self-contained (one `.so`, no external libcimba to ship).

## Prerequisites

`uv` provides Python 3.13, `meson`, `ninja`, `cffi`, and `meson-python`. You
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
uv run pytest                 # smoke tests
uv run python examples/demo_mg1_simapi.py   # SimPy-flavored M/G/1 demo
```

No manual `.venv` activation needed — `uv run` handles it. The first `uv sync`
compiles the C library + cffi extension; later runs are incremental.

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
