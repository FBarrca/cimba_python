# Missing Python API Features

This file tracks native Cimba documentation features that are preserved in the
Python docs for conceptual parity but are not exposed by the current Python API.

## Native Timeseries And Text Reports

- Raw `cmb_timeseries`, `cmb_datasummary`, `cmb_wtdsummary`, histogram,
  correlogram, and native report-printing APIs are not exposed.
- Python exposes summary accessors such as `sim.mean_level()`,
  `sim.mean_in_use()`, `sim.pool_mean_in_use()`,
  `sim.store_mean_length()`, `sim.pq_mean_length()`, and dataset accessors.

## Alias Tables

- Reusable native Vose alias-table objects (`cmb_random_alias_create()`,
  `cmb_random_alias_sample()`, `cmb_random_alias_destroy()`) are not exposed.
- Use `sim.categorical()` or `sim.loaded_dice()` for weighted discrete sampling.


## Resource Guard Observer Registration

- Direct `cmb_resourceguard_register()` observer wiring is not exposed.
- Python exposes conditions through `sim.Condition`, `sim.Predicate`,
  `@model.predicate`, `sim.wait_for()`, and `sim.signal()`.

## CUDA And Hardware Hooks

- CUDA integration, explicit GPU stream assignment, and hardware-in-the-loop
  hooks are not exposed by the Python API.

## Thread Count Control

- `cimba.use_threads(n)` currently reports the number of logical CPUs available
  and accepts `n` for API compatibility, but it does not yet configure the native
  worker pool.

## API Reference Generation

- The Python docs use Sphinx autodoc for `cimba` and `cimba.sim`.
- The copied `Doxyfile` files are placeholders because Doxygen/Exhale are only
  needed for the upstream C API reference.
