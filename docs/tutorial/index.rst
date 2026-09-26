.. _tutorial:

Tutorial
========

This tutorial is a guided tour. Each chapter builds one model, introduces a
handful of new ideas, and explains *why* the code is shaped the way it is. By
the end you will have used every major feature of Cimba Python on problems
that look like real work.

The chapters follow the upstream `Cimba C tutorial`_, so if you know the C
library you will recognize the models. Every chapter has a matching,
self-contained script in the repository's ``tutorial/`` directory. The code on
these pages is pulled straight from those scripts, so what you read is exactly
what runs.

.. code-block:: bash

   uv run python tutorial/tut_1_1.py

.. list-table::
   :header-rows: 1
   :widths: 8 32 60

   * - Ch.
     - Model
     - What you learn
   * - :doc:`1 <mm1>`
     - A single-server queue
     - Models, inputs, processes, entities, outputs, the measurement window,
       logging, datasets, captures, composition, sweeps and parallel runs.
       Ends by driving the same queue with recorded data.
   * - :doc:`2 <cheese>`
     - Mice, rats and a cat
     - Resources with capacity, priorities, preemption, interrupts, signals,
       process handles and model-internal randomness.
   * - :doc:`3 <park>`
     - An amusement park
     - Dynamic models (``spawn``/``release``), priority stores, process
       timers, ``suspend``/``resume``, and balking, reneging and jockeying.
   * - :doc:`4 <harbor>`
     - A weather-gated harbor
     - Composing several models, references between siblings, conditions and
       predicates, process priorities, and comparing designs fairly.
   * - :doc:`5 <assembly>`
     - An assembly line
     - Handing model objects between stores, captured histories, plotting,
       structure diagrams and finding a bottleneck.
   * - :doc:`6 <inventory>`
     - A multi-echelon supply chain
     - Data-driven simulation: time-indexed ``Series`` inputs, joint
       bootstraps, checking an input model before trusting it, provenance, and
       comparing input models head to head.

If you only have half an hour, read chapter 1 and then chapter 6.

.. toctree::
   :maxdepth: 2
   :hidden:

   mm1
   cheese
   park
   harbor
   assembly
   inventory
