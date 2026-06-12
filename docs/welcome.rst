.. _welcome:

Cimba Python - Multithreaded Discrete Event Simulation
======================================================
Cimba Python is a fast discrete event simulation package wrapping the native
Cimba library written in C and assembly, providing process-oriented simulation
combined with multithreaded parallelism for high performance on modern CPUs.

Parallelizing discrete event simulation is both a very hard and a trivially
simple problem, depending on the way you look at it. Parallelizing a single
simulation run is near impossible, since all events and processes inside the
simulated world depend on a shared time variable and cannot race ahead.

Luckily, we almost never run only a single simulation run, but a possibly large
experiment consisting of many trials (replications and parameter combinations)
to generate statistical results. These trials are *intended* to be independent,
making them near-trivial to parallelize by simply running them all at the same
time, or at least running as many as you have CPU cores available for.

Why should I use Cimba Python?
------------------------------
It is fast, powerful, reliable, and free.

* *Fast*: The speed from multithreaded parallel execution translates to high
  resolution in your simulation modelling. You can run hundreds of replications
  and parameter variations in just a few seconds, generating tight confidence
  intervals in your experiments and high density of data points along parameter
  variations. Python process bodies are compiled with Numba and call into the
  native Cimba engine.

* *Powerful*: Cimba Python provides a compact ``cimba.sim`` modeling API for
  well-engineered discrete event simulation models. Processes are plain Python
  functions registered with ``@model.process``. They block with calls like
  ``sim.hold()``, ``sim.get()``, and ``sim.acquire()`` without using ``yield``.

  The Python API exposes Cimba processes, buffers, resources, resource pools,
  object stores, priority queues, condition variables, datasets, and a wide
  range of fast random number generators. Python models declare parameters,
  outputs, state, and simulation entities as annotated fields on a
  ``sim.Model`` subclass, and the package generates the native trial function.

  Some lower-level C extension points are not exposed in Python yet. These are
  tracked in :doc:`missing_features`.

* *Reliable*: Cimba is well engineered, self-contained open source. There is no
  mystery to the results you get. The native library is written with liberal use
  of assertions to enforce preconditions, invariants, and postconditions in each
  function. The Python package adds smoke tests that build and execute models
  through the public Python API.

* *Free*: Cimba Python should fit well into the budget of most research groups.

What can I use Cimba Python for?
--------------------------------
It is a general purpose discrete event simulation package, in the spirit of a
21st century descendant of Simula67. It may be the right tool for the job if you
need quantitative performance analysis of some system that is so complex that it
is not possible to derive an analytical solution, but where the behavior and
interactions of the constituent parts can be described in Numba-compilable
Python code.

For example, you can use it to model:

* computer networks,

* hospital patient flows,

* transportation networks,

* operating system task scheduling,

* manufacturing systems and job shops,

* military command and control systems,

* queuing systems like bank tellers and store checkouts,

* urban systems like emergency services and garbage collection,

* ...and many other application domains along similar lines.

See :ref:`the tutorials <tutorial>` for illustrations of both expressive power
and how to use it for multi-threaded computing power.

If you look under the hood, you will also find the native Cimba components like
stackful coroutines, fast memory pool allocators, and sophisticated data
structures. The Python wrapper exposes the stable modeling layer and leaves some
of the C-native internals intentionally hidden until a safe Python API exists.

How can I get it?
-----------------
You clone the Python wrapper repository, initialize the Cimba submodule, build,
and test it with ``uv``. You will need a C build chain, NASM, git, and the Python
build tooling installed by ``uv``.

See also the :ref:`installation guide <installation>` for a more detailed
description.
