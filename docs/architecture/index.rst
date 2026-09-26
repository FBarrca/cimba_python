Architecture
============

This section explains how Cimba Python is built: what each part is
responsible for, how a model class becomes native code, how input data
reaches a running trial, and which design decisions hold it all together. You
don't need any of this to *use* the library. It's for contributors, for the
curious, and for anyone who wants to know why the API looks the way it does.

In one sentence: Cimba Python is **a compiler for model classes, an
input-modeling layer, and a native trial runner, all in one library and one
process**. There are no services, no plugins and no background processes.

The system in layers
--------------------

The picture has too many parts to read at once, so we look at it three ways.
First, the whole system as layers. Each layer uses the ones below it. The
two orange boxes are the **input-modeling pillar**: ``cimba.inputs`` on the
Python side and ``inputs.c`` in the native runtime.

.. raw:: html
   :file: ../static/diagrams/layers.svg.html

.. rst-class:: cdiag-caption

   Click any box to jump to its description. The dashed boxes are your code.

**Your code** declares model classes, configures objects (sweeps, sources,
captures), defines experiments and supplies observed data.

**The public API** is small. ``cimba`` re-exports the modeling vocabulary and
the experiment and results types. ``cimba.inputs`` is the catalogue of
sources. ``cimba.analysis``, ``cimba.random`` and ``cimba.diagrams`` cover
statistics, in-model randomness and structure.

**The model core** is pure Python: ``modeling`` defines the
language, ``schema`` describes classes and configured trees, ``layout`` turns
them into C-compatible memory layouts, and ``results`` holds outcomes.

**Compile & run** is where Python meets native code: ``compiler``
turns each class into native callbacks with Numba, ``experiments`` binds
configured objects to trial memory and runs them, and ``engine`` is the single
``ctypes`` boundary.

**The native runtime** is one C library, ``libcimba_py``. It contains the
Cimba engine (stackful coroutines, event queue, entities), a descriptor-driven
trial runner (``trial.c``), the input runtime (``inputs.c``) and the verbs
that compiled model code calls (``verbs.c``, ``nbshim.c``).

Two paths that meet in ``experiments``
--------------------------------------

The second view follows the data. A model **class** and a configured
**object** take different routes through the packages. Classes go
``schema → compiler`` and are compiled once. Configured objects go
``schema → layout`` and are laid out per experiment. Observed data enters
through ``cimba.inputs``. All three converge in ``experiments``:

.. raw:: html
   :file: ../static/diagrams/pipeline.svg.html

That separation is the core architectural decision: **compiled code depends
only on classes**. Structure, parameter values and input sources are data,
bound at run time. :doc:`pipeline` walks through it.

Zooming into the input pillar
-----------------------------

The third view zooms into the input-modeling pillar. Any of four sources can
feed the same model field without changing the model code:

.. raw:: html
   :file: ../static/diagrams/input_pillar.svg.html

Distribution sources are drawn inside the trial. Recorded traces are replayed
as they are. Bootstrap and fitted-model rows are generated in Python once per
replication and shared by every design point. Whichever source you bind,
``inputs.c`` gives the input its own random stream and applies its exhaustion
policy. So you can swap sources, or sweep them against each other, without
touching the model or recompiling. :doc:`inputs` has the details.

Reading order
-------------

.. list-table::
   :widths: 30 70

   * - :doc:`packages`
     - Each package's responsibility and the enforced dependency rules.
   * - :doc:`pipeline`
     - From class to native callbacks, from objects to trial memory, and
       what happens inside ``run()``.
   * - :doc:`inputs`
     - The input runtime: slots, streams, series buckets, row generation,
       exhaustion and extension, provenance.
   * - :doc:`native`
     - The C library: trial lifecycle, abandonment, spawned models,
       captures, the ABI and the ``ctypes``/Numba boundaries.
   * - :doc:`principles`
     - Principles, invariants and the decisions behind them, with the
       alternatives that were rejected.

To read the source in dependency order, follow the packages bottom-up:
``modeling → inputs → schema → layout → compiler → experiments → results →
analysis``.

.. toctree::
   :hidden:

   packages
   pipeline
   inputs
   native
   principles
