# Missing Python API Features

This file mirrors `docs/missing_features.rst` for quick repository browsing. It
tracks Python binding gaps using the public Python API vocabulary.

## Reusable Weighted Samplers

- Process bodies can draw weighted discrete values with `sim.categorical()` and
  `sim.loaded_dice()`.
- A reusable Python object for precomputed weighted sampling tables is not
  exposed yet.

## Custom Resource Guard Observers

- Python exposes queues, resources, pools, stores, priority queues, and
  conditions as the public waiting mechanisms.
- Custom observer registration on the shared waiting machinery is not exposed
  yet.

## External Compute Hooks

- Explicit external accelerator and hardware-in-the-loop hooks are not exposed
  by the Python API yet.

## Thread Count Control

- `Experiment.run()` currently uses the package's default parallel execution
  policy. A public Python option for choosing the worker count explicitly is not
  exposed yet.

## API Reference Generation

- The Python docs use a lightweight Sphinx page for `cimba` and `cimba.sim`.
- More detailed generated documentation for every public Python helper can be
  added once the import-time documentation build is stable across supported
  platforms.
