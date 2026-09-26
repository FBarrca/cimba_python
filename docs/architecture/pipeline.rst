From classes to trials
======================

This page follows a model from its class definition to native trials and
back to results.

Compile classes, bind instances
-------------------------------

The organizing rule of the implementation:

   **Compiled code is a function of a model class. Everything about a
   particular configured model (its structure, values and input sources) is
   data, bound into memory at run time.**

That rule is why you can rewire a network, resize a list, sweep a parameter
or swap a trace for a bootstrap without recompiling. It's also why errors in
model code are reported per class and method, and why the compiler has no
access to instances or sources (a rule enforced by the test suite).

The class path
--------------

1. **Schema.** ``ClassSchema.of(cls)`` reads the class's annotations and
   decorated methods through its whole hierarchy and produces a frozen
   description: fields with kinds and types, processes with copies and
   priorities, hooks, predicates and events. It is cached per class.

2. **Record layout.** ``RecordLayout.of(schema)`` turns the fields into a
   C-compatible record, a NumPy structured dtype:

   .. code-block:: text

      Record for class StockingFacility
      ┌───────────────────────────────────────────────┐
      │ class_descriptor*   → dispatch table (events, │  first member of every record
      │                        predicates, functions) │
      │ network*            Ref["MultiEchelon…"]      │  pointer
      │ node                int64                     │  Param
      │ upstream*           Ref | None                │  pointer (may be 0)
      │ base_stock          float64                   │  Param
      │ …                                             │
      │ on_hand             float64                   │  State
      │ service_level       float64                   │  Output
      │ orders              → store handle            │  entity (filled by trial.c)
      │ demand              InputSlot (fixed size)    │  Series
      └───────────────────────────────────────────────┘

   A subclass's record contains all of its base class's fields, so code that
   sees a record through the base class reads the same field names.

3. **Views and handles.** The compiler registers the record type with Numba
   as a *view*: attribute access compiles to a load or store at a fixed
   offset. Child and reference fields compile to typed pointer loads, with a
   null check that abandons the trial. Optional references become Numba
   optionals. Lists become bounds-checked (pointer, length) views. Entity
   and input fields get methods (``put``, ``get``, ``next``, ``now`` …)
   through Numba overloads, each lowering to one call into the native
   library. Because these are real types, misuse is a compile-time error.

4. **Compile.** ``compiler.ensure(schema)`` compiles each process, hook,
   predicate, event and function with ``numba.njit`` for the record type,
   wraps each in a C-callable ``cfunc``, and builds the class's dispatch
   table. Table slots follow ``ClassSchema.slots``: a subclass keeps its
   bases' slots in the same positions and appends its own. A call
   ``view.f(x)`` compiles to "load the table pointer from the record, load
   slot *k*, call". The *record's* class picks the implementation, which is
   how overrides reach callers that only know the base class. The result,
   a ``CompiledClass``, is cached for the life of the process. Errors are
   rethrown as ``ModelCompileError`` with ``file:line``.

Nothing in this path looks at the model *objects*: the key is the class.

The instance path
-----------------

1. **Assembly.** When you create an ``Experiment``, ``Assembly.of(root)``
   walks the configured tree and takes a snapshot: every static instance
   with its canonical label (``network.facilities[2]``), its schema and its
   resolved field values. Entities are materialized, lists frozen, sweeps
   and sources recorded as values. This is where missing parameters or
   sources, dangling references and reused instances are reported.

2. **Trial image layout.** ``TrialImageLayout.of(assembly)`` places every
   instance's record inside one contiguous **trial block**:

   .. code-block:: text

      One block per trial (a run allocates an array of equal-size blocks)
      ┌─────────────────────────────────────────────────────────────┐ +0
      │ TrialHeader: seed · trial index · window · status ·         │
      │              error[256] · capture* · descriptor*            │
      ├─────────────────────────────────────────────────────────────┤
      │ record  network                  (root)                     │
      │ record  network.facilities[0]    SourceFacility             │
      │ record  network.facilities[1]    StockingFacility           │
      │ …                                                           │
      └─────────────────────────────────────────────────────────────┘
      Shared, read-only, per run:
        trial descriptor → entity, process, hook and input tables
        class dispatch tables · input row buffers · trace arrays
      Per trial, on the side:
        spawned model records (freed when the trial ends)

3. **Design.** ``Design.of(assembly)`` finds every sweep, including sweeps
   inside distribution parameters and sweeps of sources, and expands them
   into design points. Each point binds every (instance, field) to a concrete
   value or source.

Where the paths meet: inside ``run()``
--------------------------------------

.. code-block:: text

   main thread (Python)                                   worker threads (C)
   ────────────────────                                   ──────────────────
   ensure CompiledClass for each class in the tree
   build descriptor tables (entities, processes with callback
     addresses, hooks bottom-up, inputs)
   plan chunks of replications within input_memory
   for each chunk:
     derive trial seeds  (SeedSequence([seed, r]) or [seed, r, point])
     generate rows for every row source, once per replication,
       vectorized; shared by all design points
     allocate trial blocks; fill headers, params, constants,
       initial state, references, list tables, input slots
     cpy_run(blocks, n, size, workers) via ctypes, GIL released ──►  per trial:
                                                                      create entities
                                                                      on_start hooks
                                                                      start processes
                                                                      window events
                                                                      event loop ⇄ model code
                                                                           ⇄ verbs / inputs.c
                                                                      on_end hooks
                                                                      copy captures
                                                                      tear down, status = OK
                                                                    on engine error:
                                                                      abandon, record reason,
                                                                      reclaim everything
     trials with an exhausted extendable input:
       regenerate that row 2× longer, reset the block, rerun (≤ 6×)
     copy outputs, cursors, captures, statuses out of the blocks
   assemble immutable Results (joined across chunks)

The same flow as a sequence diagram:

.. mermaid::

   sequenceDiagram
     participant U as your code
     participant E as experiments
     participant C as compiler
     participant I as cimba.inputs
     participant N as libcimba_py
     U->>E: Experiment(model, ...).run()
     E->>C: ensure(ClassSchema) per class
     C-->>E: CompiledClass (cached)
     loop each chunk of replications
       E->>I: generate(rngs, length) per row source
       I-->>E: rows, one per replication
       E->>E: allocate and bind trial blocks
       E->>N: cpy_run(blocks, workers), GIL released
       N-->>E: statuses, outputs, cursors, captures
       opt a trial exhausted an extendable input
         E->>I: generate(rngs, 2 × length)
         E->>N: rerun that trial
       end
     end
     E-->>U: Results

A few properties fall out of this design:

* **Workers never touch Python.** Everything a trial needs is in its block
  or in shared read-only tables. Trials run in any order on any thread.
* **Results don't depend on execution strategy.** Seeds depend only on
  (seed, replication[, point]); rows only on (trial seed, tag); extension
  reproduces prefixes exactly. So worker count, chunk size and extension
  don't change any number.
* **One host call per chunk.** Python calls the native runner once and
  gets control back when every trial of the chunk has finished.

Back to Python: results
-----------------------

After the native call, ``experiments`` reads each instance's output columns
out of the blocks (masking failed trials with ``nan``), each input slot's
cursor, the extension counts and the detached capture arrays. It then builds
a ``Results`` keyed by the model objects themselves. An instance index maps
each object to its record offset, so ``results[dc]`` needs no string
parsing. Input rows are *not* copied into the results; ``rows(p, r)``
regenerates them from the seed and source when asked.
