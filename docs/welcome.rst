.. _welcome:

Cimba Python - Multithreaded Discrete Event Simulation
======================================================

Cimba Python is a fast process-oriented discrete event simulation package for
Python. Models are written with ``sim.Model`` declarations and ordinary Python
process functions; experiments run many independent trials across available CPU
cores.

Parallelizing a single simulated world is difficult because all processes share
one simulated clock. Parallelizing an experiment is much simpler: replications
and parameter combinations are intended to be independent, so they can be run at
the same time and summarized afterwards.

Why should I use Cimba Python?
------------------------------

It is fast, powerful, reliable, and free.

* *Fast*: Independent trials run in parallel, and process bodies use the
  compiled Cimba Python simulation layer. That makes high replication counts and
  dense parameter sweeps practical.

* *Powerful*: The ``cimba.sim`` API includes processes, queues, resources,
  resource pools, stores, priority queues, conditions, timers, events, datasets,
  random distributions, logging helpers, and experiment tables.

* *Reliable*: Models are explicit. Parameters, outputs, state, and simulation
  entities are declared on the model class, and invalid declarations or failed
  trials are surfaced clearly.

* *Free*: Cimba Python is open source and intended to fit research,
  engineering, and teaching budgets.

What can I use Cimba Python for?
--------------------------------

Cimba Python is a general-purpose discrete event simulation package. It is a
good fit when a system is too complex for a closed-form analytical solution,
but the entities, resources, timing, randomness, and interactions can be
described as Python model code.

For example, you can use it to model:

* computer networks,
* hospital patient flows,
* transportation networks,
* operating system task scheduling,
* manufacturing systems and job shops,
* military command and control systems,
* queuing systems like bank tellers and store checkouts,
* urban systems like emergency services and garbage collection,
* and many other systems with active agents and limited resources.

See :ref:`the tutorials <tutorial>` for examples that move from a simple queue
to resources, process interruptions, dynamic agents, conditions, and parallel
experiments.

How can I get it?
-----------------

Clone the Python wrapper repository, initialize its submodules, and install the
project environment with ``uv``:

.. code-block:: bash

    git clone <repo-url> cimba_python
    cd cimba_python
    git submodule update --init --recursive
    uv sync

See the :ref:`installation guide <installation>` for platform details,
verification commands, wheel builds, and troubleshooting.
