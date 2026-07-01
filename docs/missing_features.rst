Missing Python API Features
===========================

This page tracks Python binding gaps. It intentionally describes them in terms
of the public Python API rather than lower-level implementation names.

Reusable Weighted Samplers
--------------------------

Process bodies can draw weighted discrete values with ``sim.categorical()`` and
``sim.loaded_dice()``. A reusable Python object for precomputed weighted
sampling tables is not exposed yet.

Custom Resource Guard Observers
-------------------------------

Python exposes the standard waiting mechanisms through queues, resources, pools,
stores, priority queues, and conditions. Custom observer registration on the
shared waiting machinery is not exposed yet.

External Compute Hooks
----------------------

Explicit external accelerator and hardware-in-the-loop hooks are not exposed by
the Python API yet. Models should keep heavy numeric work behind Python
functions with clear numeric inputs and outputs.

Thread Count Control
--------------------

``Experiment.run()`` currently uses the package's default parallel execution
policy. A public Python option for choosing the worker count explicitly is not
exposed yet.

API Reference Generation
------------------------

The Python docs use a lightweight Sphinx page for ``cimba`` and ``cimba.sim``.
More detailed generated documentation for every public Python helper can be
added once the import-time documentation build is stable across supported
platforms.
