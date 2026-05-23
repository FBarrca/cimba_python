Native Cimba Concepts
=====================

Cimba Python is a binding layer over the Cimba C library. The native engine owns
the event queue, process switching, resources, random number generation, and
statistics objects.

What the Python layer adds
--------------------------

The binding layer adds:

* Python classes around native Cimba objects
* Python process callbacks
* Python objects in object queues and priority queues
* context-managed simulation ownership
* type stubs for editor support
* a wheel build that embeds the native library

What the C docs explain best
----------------------------

Read the `Cimba C background`_ for:

* why Cimba uses stackful coroutines
* how the dispatcher and simulated time interact
* how process-oriented modeling relates to Simula-style coroutines
* why replications are the natural parallelization boundary

Read the `Cimba C API reference`_ for native object contracts and lower-level
function details.
