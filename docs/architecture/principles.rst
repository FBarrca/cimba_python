Principles and decisions
========================

This page records *why* Cimba Python is shaped the way it is: the principles
the design follows, the invariants every change must keep, and the main
decisions with the alternatives that were rejected.

Principles
----------

1. **Model the domain, not the ABI.** Every user-visible concept is a
   discrete-event or input-modeling concept: model, process, entity, input,
   source, sweep, trial. Nothing in the API exists only to satisfy the C
   layer.
2. **Inputs are first-class and source-agnostic.** Observed real-world
   variability is a declared input. Its source is configuration,
   interchangeable without touching model code, with its own random stream
   and recorded provenance.
3. **The running system mirrors the object graph.** One record per model
   instance; references are pointers; lists are pointer tables.
4. **Compile classes, bind instances.** Code depends only on class
   definitions. Structure, values and sources are data.
5. **Let the compiler compile.** The modeling language is implemented in the
   JIT's type system (views, handles, overloads), not by rewriting source
   text.
6. **Fixed machinery is native and precompiled.** Only user methods are
   JIT-compiled. The trial lifecycle and the input runtime are C.
7. **Configure objects; read results through objects.** No string addressing.
   Experiments snapshot configuration, and results are immutable.
8. **One way to do each thing; boundaries enforced by tests.** Package
   rules live in the test suite, not in a style guide.

Invariants
----------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Invariant
     - Statement
   * - Trial independence
     - A trial reads shared read-only data and writes only its own block and
       its own native objects.
   * - Reproducibility
     - A trial's outcome is a function of (model classes, configuration,
       design point, trial seed). It is independent of worker count,
       scheduling, memory chunking and input extension.
   * - Structural immutability
     - The instance tree, fields and input declarations are fixed for an
       experiment. Experiments vary values and sources only.
   * - Window semantics
     - Time-weighted statistics cover the measurement window. Datasets are
       cleared when it opens.
   * - Lifecycle pairing
     - Every native object is created and destroyed exactly once per trial,
       including abandoned trials.
   * - Hook order
     - Static ``on_start`` and ``on_end`` hooks run bottom-up.
   * - Objects are identity
     - Users address everything through objects; canonical paths are labels.
   * - Source independence
     - Changing a source changes values only, never compiled code or other
       inputs' streams.
   * - Stream synchronization
     - An input's values in a trial depend only on the trial seed, the
       input's identity and its source.
   * - Prefix stability
     - Regenerating a row source with more values reproduces the leading
       values, so extension can't change results.

Key decisions
-------------

How inputs reach model code
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 19 19

   * -
     - **Source-agnostic input fields** (chosen)
     - Trace fields + inline random calls
     - Resampling verbs inside the trial
     - Python callback per value
   * - Swap distribution ↔ trace ↔ bootstrap
     - configuration only
     - rewrite the model
     - partly
     - configuration
   * - Per-input streams (CRN)
     - yes
     - one shared stream
     - shared stream
     - up to the user
   * - Block, joint, model-based structure
     - yes (host rows)
     - yes
     - i.i.d. only, realistically
     - yes
   * - Speed
     - one C call
     - one C call
     - one C call
     - a GIL round trip per value
   * - Running out
     - explicit policies, exact extension
     - silent
     - n/a
     - n/a

Runtime data model
~~~~~~~~~~~~~~~~~~

**Per-instance records, compiled per class** (chosen) versus one flattened
record per model. Flattening gives constant offsets, but it leaks flattened
names into the API, needs slot and owner tables to rebuild the object graph,
and forces recompilation whenever structure changes. Per-instance records
cost one pointer load to cross between models and make structure pure data.

Compilation strategy
~~~~~~~~~~~~~~~~~~~~

**Numba extension types** (chosen) versus rewriting model source into
flattened functions. With typed views and handles, real type inference sees
your code as written. Misusing a handle is a compile error with a line
number, not a runtime crash, and callbacks don't need inspectable source.
Generating C code was rejected as a second compiler to maintain.

Native binding
~~~~~~~~~~~~~~

**ctypes plus direct symbol calls** (chosen) versus Cython or cffi. The host
needs about ten calls, so ctypes costs nothing measurable. It keeps the C
library free of the Python C API, which gives one wheel per platform instead
of one per CPython version. Model code calls the same library directly
through registered symbols.

Isolation
~~~~~~~~~

**In-process trials with engine-level abandonment** (chosen) versus a process
per trial. Abandonment handles every engine-detected error without leaking.
Typed handles and bounds checks prevent memory errors in model code.
Process isolation would add a second execution path and serialization costs
for the rare remaining crash.

Results format
~~~~~~~~~~~~~~

**NumPy arrays in immutable containers** (chosen) versus pandas or xarray.
Arrays are the natural shape for ``(points, replications)`` data and add no
dependency; ``to_table()`` feeds pandas when wanted.

Common random numbers by default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sweeps exist to compare configurations and input models. Common seeds per
replication, combined with per-input streams, make those comparisons paired
and sharp. ``seeding="independent"`` remains available.

Current limitations
-------------------

Known gaps in 0.7.0 that the documentation works around:

* An input with ``on_exhausted="end_trial"`` stops its trial, but the trial
  is reported as failed and its outputs are ``nan``.
* ``cb.Container(initial=...)`` is accepted but ignored. Put initial stock
  in an ``@cb.on_start`` hook.
* The window stops the static model's processes but not those of spawned
  models. A spawned process that never finishes or blocks keeps the trial
  running.
* Undecorated model methods can't be called from compiled code. Use
  module-level ``numba.njit`` helpers that take the model view.
* The compile cache lives in memory, so each new Python process compiles its
  classes once.
