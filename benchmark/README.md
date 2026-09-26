# Benchmarks

This directory contains the Cimba Python M/M/1 queue benchmarks. The matching
SimPy and native C benchmark sources are vendored with the C library in
`subprojects/cimba/benchmark/`. Additional runnable tutorial models live in
[`../tutorial/`](../tutorial/).

The current measurements are summarized in the [project README](../README.md).
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
Cimba, NumPy, Numba, and LLVM metadata. Add `--check` to fail if compilation
is unsuccessful or a warm-cache run records no persistent-cache hit.
