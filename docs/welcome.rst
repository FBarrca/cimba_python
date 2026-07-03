.. _welcome:

Cimba Python - Multithreaded Discrete Event Simulation
======================================================

Cimba Python is a fast process-oriented discrete event simulation package for
Python. It gives Python models access to the native
`Cimba <https://github.com/ambonvik/cimba>`_ simulation engine, while keeping
the model itself in ordinary Python functions and ``sim.Model`` declarations.

The goal is simple: make simulation experiments feel like Python, but run the
hot simulation loop with compiled machinery underneath. You describe active
entities as processes, declare trial-local parameters and state on a model
class, and run independent replications or parameter combinations as an
experiment table.

Why simulation experiments parallelize well
-------------------------------------------

Parallelizing one simulated world is hard. All processes in that world share
one simulated clock, and the next event may change what every later event means.
Letting different parts of that same world race ahead is usually a correctness
problem waiting to happen.

Parallelizing an experiment is much cleaner. Replications and parameter
combinations are intended to be independent: one trial with one random seed and
one parameter row should not depend on another. Cimba Python uses that natural
independence. A single trial stays deterministic and sequential in simulated
time, while many trials can execute at the same time across CPU cores.

That makes Cimba Python a good fit for the work simulation modelers often
actually need to do:

* run enough replications to get useful confidence intervals,
* sweep parameter values without waiting all afternoon,
* compare scenarios with a consistent model definition,
* keep process logic readable instead of spreading it across callbacks,
* and collect scalar outputs that can be summarized, plotted, or exported.

Why should I use Cimba Python?
------------------------------

It is fast, expressive, explicit, and open source.

* *Fast*: Process bodies use Cimba Python's compiled simulation layer, and
  independent trials can run in parallel. This makes high replication counts,
  dense parameter sweeps, and hot process loops much more practical than with
  pure Python event scheduling alone.

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

The common pattern is the same across these domains: active entities become
processes, passive constraints become model fields, randomness comes from
``cimba.sim``, and the experiment table turns model runs into data.

When is it not the right tool?
------------------------------

Cimba Python is aimed at process-oriented discrete event simulation. It may be
more machinery than you need for a one-off deterministic calculation, a small
spreadsheet-style Monte Carlo, or a model where pure vectorized NumPy already
does all the work. It is most useful when behavior unfolds through events,
waiting, resources, queues, and many independent trials.

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

Read :ref:`the background section <background>` if you want to understand why
Cimba Python is structured this way: process-oriented simulation, event queues,
resources, random draws, datasets, and parallel trial execution.

Use :doc:`api/cimba` as a map of the public Python surface while writing code.
