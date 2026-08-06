# Benchmarks

This directory contains the Cimba Python M/M/1 queue benchmarks. The matching
SimPy and native C benchmark sources are vendored with the C library in
`subprojects/cimba/benchmark/`. Additional runnable tutorial models live in
[`../tutorial/`](../tutorial/).

On an AMD Ryzen 7 9700X under WSL Ubuntu 24.04, averaged over 10 runs, with
Cimba Python timed after its one-time Numba compile:

| Benchmark | SimPy | Cimba Python | Cimba C |
| --- | ---: | ---: | ---: |
| Single core, single trial | 2.612 s | 0.096 s | 0.083 s |
| Multicore, 100 trials | 36.807 s | 1.131 s | 0.970 s |

The benchmark data and charts are in
[`AMD_Ryzen_7_9700X_WSL.ods`](AMD_Ryzen_7_9700X_WSL.ods).

Run the current Python benchmarks from the repository root:

```bash
uv run python benchmark/mm1.py
uv run python benchmark/mm1_multi.py
```

## Component compilation

`component_compilation.py` measures compilation in fresh Python interpreters.
It reports median and median absolute deviation for import, model definition,
model construction/precompilation, first experiment, and cached experiment.
The two tutorial models are complemented by synthetic component collections at
1, 10, 100, and 1,000 instances; the synthetic model includes indexed and
structured processes, a spawnable process, predicates, events, and
multi-instance collectors.

```bash
uv run python benchmark/component_compilation.py
uv run python benchmark/component_compilation.py --cache cold --json cold.json
uv run python benchmark/component_compilation.py --cache warm --json warm.json
```

Every JSON result contains the raw samples and Python, operating-system, CPU,
Cimba, NumPy, Numba, and LLVM metadata. CI runs a two-process warm-cache smoke
benchmark natively on Linux, Windows, and Apple silicon; `--check` makes that
job fail if compilation is unsuccessful or the second process records no
persistent-cache hit.
