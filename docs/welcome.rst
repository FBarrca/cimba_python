.. _welcome:

Cimba Python - Process-Oriented Discrete Event Simulation
=========================================================

Cimba Python is a fast process-oriented discrete event simulation package for
Python. It gives Python models access to the native
`Cimba <https://github.com/ambonvik/cimba>`_ simulation engine, while keeping
the model itself in ordinary Python functions and ``sim.Model`` declarations.

The goal is simple: make simulation experiments feel like Python, but run the
hot simulation loop with compiled machinery underneath.

Why should I use Cimba Python?
------------------------------

It is fast, expressive, explicit, and open source.

* *Fast*: Process bodies use Cimba Python's compiled simulation layer. This
  makes high replication counts, dense parameter sweeps, and hot process loops
  much more practical than with pure Python event scheduling alone.

* *Expressive*: Processes are ordinary Python functions that call blocking
  simulation operations such as ``sim.hold()``, ``sim.get()``,
  ``sim.acquire()``, or ``sim.wait_for()``. There is no ``yield`` protocol to
  thread through every helper function.

* *Experiment-oriented*: Parameters, outputs, state, and simulation entities
  are declared on a ``sim.Model`` subclass. ``model.experiment(...)`` creates a
  table of independent trials, and ``Experiment.run()`` executes the table.

* *Well-equipped*: The ``cimba.sim`` API includes processes, queues, buffers,
  stores, priority queues, resources, resource pools, conditions, timers,
  events, datasets, time series summaries, random distributions, logging
  helpers, and dynamic process spawning.

* *Observable*: Models can log process activity, record queue/resource
  histories, tally datasets, and collect named outputs from every trial. That
  makes it easier to debug a model and easier to turn results into analysis.

* *Explicit*: Invalid declarations and failed trials are surfaced clearly.
  Trial-local model fields make it visible which pieces of state belong to the
  simulated world and which are external Python analysis code.

Benchmark snapshot
------------------

The included benchmark is an M/M/1 queue with one million arrivals per trial.
On an AMD Ryzen 7 9700X under WSL Ubuntu 24.04, averaged over 10 runs, with
Cimba Python timed after its one-time Numba compile:

.. list-table::
   :header-rows: 1

   * - Benchmark
     - SimPy
     - Cimba Python
     - Cimba C
   * - Single core, single trial
     - 2.612 s
     - 0.096 s
     - 0.083 s
   * - Multicore, 100 trials
     - 36.807 s
     - 1.131 s
     - 0.970 s

The benchmark data and charts are in
``benchmark/AMD_Ryzen_7_9700X_WSL.ods``. Benchmarks depend on the model,
machine, Python version, and build configuration, so treat these numbers as a
reason to measure your own model rather than as a universal constant.

What can I use Cimba Python for?
--------------------------------

Cimba Python is a general-purpose discrete event simulation package. It is a
good fit when a system is too complex for a closed-form analytical solution,
but the entities, resources, timing, randomness, and interactions can be
described as Python model code.

For example, you can model:

* computer networks and distributed services,
* hospital patient flows and emergency department triage,
* transportation terminals, ports, and rail yards,
* operating system task scheduling,
* manufacturing systems, machine breakdowns, and job shops,
* military command, control, sensor, and logistics systems,
* call centers, bank tellers, store checkouts, and other queueing systems,
* urban systems like emergency services, waste collection, or maintenance,
* and other systems with active entities competing for limited resources.

These domains share one modeling pattern. Active entities are written as
processes, the constraints they compete over are declared as model fields, and
randomness is drawn from ``cimba.sim``. Running the model then produces the data
you analyze.

How can I get it?
-----------------

Install from Python packaging tools:

.. code-block:: bash

    pip install cimba

or with ``uv``:

.. code-block:: bash

    uv add cimba

Python 3.13 or newer is required. The wheel embeds the native Cimba library, so
you do not need to install Cimba separately.

For development from a source checkout:

.. code-block:: bash

    git clone <repo-url> cimba_python
    cd cimba_python
    git submodule update --init --recursive
    uv sync
    uv run pytest

See the :ref:`installation guide <installation>` for platform details,
verification commands, wheel builds, and troubleshooting.

Where should I start?
---------------------

Start with :doc:`concepts/index` if you want the modeling vocabulary first:
``sim.Model``, trial-local ``env``, process functions, shared entities,
experiments, and outputs.

Then read :ref:`the tutorial <tutorial>` when you want to build complete
models. It moves from a simple queue to parallel experiments, resources,
process interruptions, dynamic agents, conditions, and a larger harbor example.

Use the :doc:`API reference <api_reference/index>` while writing model code.
