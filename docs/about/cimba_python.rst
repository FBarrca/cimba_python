Cimba Python
============

Cimba Python is a binding layer for the Cimba C simulation engine. It is built
with Cython and meson-python and uses the vendored C library in
``subprojects/cimba``.

The package currently wraps the core single-thread simulation API:

* simulations and event queues
* stackful Cimba processes
* buffers, object queues, and priority queues
* resources, resource pools, and conditions
* random distributions
* datasets, time series, and summary helpers

Parallel experiment orchestration is planned as a later Python layer. Today,
write a function that runs one trial and call it repeatedly from Python.
