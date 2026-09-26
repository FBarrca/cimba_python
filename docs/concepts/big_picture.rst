The big picture
===============

Two worlds
----------

Everything in Cimba Python happens in one of two worlds, and knowing which
one you are in explains almost every rule in the library.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - **The host world** (ordinary Python)
     - **The trial world** (compiled, native)
   * - Code
     - Your script, ``__init__`` methods, sweeps, source construction,
       analysis.
     - Methods marked ``@cb.process``, ``@cb.on_start``, ``@cb.on_end``,
       ``@cb.predicate``, ``@cb.event``.
   * - ``self`` is…
     - the Python object you created.
     - a typed view of *one trial's* copy of the model data.
   * - Runs…
     - once, in the main thread.
     - once per trial, on a worker thread, with no interpreter.
   * - Can use…
     - anything in Python.
     - numbers, NumPy, loops, Numba-compatible helpers, and the Cimba verbs
       (``hold``, ``get``, ``next`` …).
   * - Time is…
     - wall-clock time.
     - simulated time, advanced by the event loop.

.. mermaid::

   flowchart LR
     subgraph host["Host world (Python)"]
       direction TB
       A["declare classes"] --> B["configure objects<br/>sweeps · sources · captures"]
       B --> C["Experiment(...)"]
       R["Results"] --> S["cimba.analysis"]
     end
     subgraph trial["Trial world (native)"]
       direction TB
       T1["trial (0, 0)"]
       T2["trial (0, 1)"]
       T3["trial (p, r) …"]
     end
     C -- "compile classes<br/>lay out blocks" --> trial
     trial -- "outputs · inputs · captures" --> R

You **configure** in the host world and **simulate** in the trial world. The
bridge between them is the experiment: it snapshots your configured objects,
compiles the classes, lays out one block of memory per trial, and hands the
blocks to the native engine.

Five steps
----------

Every Cimba Python program follows the same five steps:

.. code-block:: text

   1. Declare      class Shop(cb.Model): fields + methods          (host)
   2. Configure    shop = Shop(); shop.x = cb.sweep(...);           (host)
                   shop.demand = inputs.trace(data); shop.q.capture()
   3. Define       exp = cb.Experiment(shop, replications=, window=, seed=)
   4. Run          results = exp.run(workers=None)                 (trial world, in parallel)
   5. Read         results[shop].output  →  Samples (points × replications)
                   cb.analysis.summary / compare / check_input

1. **Declare** model classes. Annotated fields say what each part of the system
   *has*; decorated methods say what it *does*.
2. **Configure** instances. Build the object tree, set parameters, bind input
   sources, mark entities for capture. Anything you might want to vary becomes
   a ``cb.sweep``. Configuration is plain attribute assignment on the objects,
   with no string keys.
3. **Define** the experiment: how many replications, which measurement
   window, which seed. The experiment takes run settings only; everything
   about the model is already on the model objects.
4. **Run.** Classes are compiled (once per process), sweeps expand into design
   points, and every (design point, replication) pair becomes an independent
   trial on some CPU core.
5. **Read** results through the same objects: ``results[shop].mean_line``.
   Results are immutable.

One vocabulary
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Term
     - Meaning
   * - **Model**
     - A class describing one part of the system. The whole simulation is the
       *root* model; it may contain other models.
   * - **Field**
     - A typed, per-instance value declared by annotation: ``Param``,
       ``State``, ``Output``, ``Input``, ``Series``, ``Ref``, an entity, a
       child model, a list, or a plain constant.
   * - **Entity**
     - A native synchronization object: ``Container``, ``Store``,
       ``PriorityStore``, ``Resource``, ``Condition``, ``Dataset``.
   * - **Process**
     - A method running as a coroutine in simulated time. It can block
       anywhere.
   * - **Input**
     - A stream of real-world variability consumed by model code, declared as
       ``Input[T]`` (a sequence) or ``Series[T]`` (time-indexed).
   * - **Source**
     - What feeds an input: a distribution, a recorded trace, a bootstrap
       resample or a fitted time-series model. Chosen at configuration time,
       never in model code.
   * - **Sweep**
     - A design axis: a list of values for a ``Param``, a distribution
       parameter or an input source.
   * - **Design point**
     - One combination of sweep values.
   * - **Replication**
     - One independent repetition with its own random numbers.
   * - **Trial**
     - One design point × one replication. The unit of parallel work.
   * - **Window**
     - Warmup, measurement duration and cooldown, or "until idle".
   * - **Results**
     - Immutable outcomes, read by model object: outputs as ``Samples``,
       inputs as ``InputRecord``, captures as ``Signal``.

The guarantees
--------------

These properties hold for every experiment, and the rest of the library is
built to keep them:

* **Trials are independent.** A trial reads shared read-only data and writes
  only its own memory, so trials can run on any core in any order.
* **Trials are reproducible.** A trial's outcome depends only on the model
  classes, the configuration, the design point and the replication's seed.
  It doesn't depend on the number of workers, scheduling, memory chunking,
  or whether an input had to be extended.
* **Model code is source-agnostic.** Changing an input's source changes the
  values it produces, never the compiled code or other inputs' values.
* **Structure is fixed per experiment.** Sweeps vary values and sources, not
  which models, fields or entities exist.
* **Objects are identity; paths are labels.** You address everything
  through objects. Canonical paths such as ``network.facilities[2]`` appear
  only in messages, descriptions and provenance.
