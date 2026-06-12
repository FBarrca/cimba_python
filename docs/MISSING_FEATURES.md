# Missing Python API Features

This file tracks native Cimba documentation features that are preserved in the
Python docs for conceptual parity but are not exposed by the current Python API.

## Logging

- Python process bodies cannot call native user logger helpers yet.
- Logger flag controls such as `cmb_logger_flags_on()` and
  `cmb_logger_flags_off()` are not exposed as public Python functions.
- Python docs preserve the logging explanation, but examples that use native
  `cmb_logger_user()` are C-only for now.

## Low-Level Events

- Arbitrary user event callbacks and direct `cmb_event_schedule()` wrappers are
  not exposed.
- Python models normally use `Model.experiment(duration=...)`, `sim.hold()`,
  timers, `sim.wait_event()`, and generated lifecycle events instead.

## Native Object Lifecycle

- Manual create/initialize/start/terminate/destroy APIs for native Cimba objects
  are not public Python APIs.
- Python models declare entities on `sim.Model` subclasses and the wrapper
  generates the lifecycle plumbing.

## Derived C Structs And Process Subclasses

- The C tutorial derives `struct visitor`, `struct server`, and `struct ship`
  from `cmb_process`. Python does not expose native subclassing of `cmb_process`.
- Use `sim.Model` state fields, `sim.Processes`, process copy indexes, and
  integer identifiers instead.

## Arbitrary Pointer Payloads

- `sim.Store` and `sim.PQueues` carry opaque `int64` values in Python, not
  arbitrary C pointers or Python objects.
- Use integer IDs, model state fields, or `sim.f2i()` / `sim.i2f()` for float
  timestamp payloads.

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

## Internal `cmi_*` Building Blocks

- Internal structures such as `cmi_slist`, `cmi_list`, `cmi_hashheap`,
  `cmi_mempool`, and coroutine internals are not public Python APIs.

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
