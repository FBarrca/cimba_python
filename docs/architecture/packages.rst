Packages and dependencies
=========================

Cimba Python is one library in one process. Its packages are named after
their responsibility, and each has a narrow job. This page describes each one
and the dependency rules between them.

.. contents::
   :local:
   :depth: 1

Public and internal
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Public (stable)
     - ``cimba`` (the facade), ``cimba.inputs``, ``cimba.analysis``,
       ``cimba.random``, and the small ``cimba.diagrams``.
   * - Internal
     - ``cimba.modeling``, ``cimba.schema``, ``cimba.layout``,
       ``cimba.compiler``, ``cimba.engine``, ``cimba.experiments``,
       ``cimba.results``. Their names say what they do; their contents may
       change between releases.
   * - Native
     - ``libcimba_py``: the Cimba C engine plus ``trial.c``, ``inputs.c``,
       ``verbs.c`` and ``nbshim.c``, as one shared library with no Python
       linkage.

There are no underscore-prefixed modules: the test suite rejects them. Every
module lives in a package whose name states its role.

.. _arch-pkg-cimba:

``cimba``: the facade
---------------------

Re-exports the modeling vocabulary (``Model``, field types, entities,
decorators, verbs), ``Experiment``/``Window``, ``Results``/``Samples``/
``Signal``, and the ``inputs``, ``random`` and ``analysis`` subpackages. It
contains no logic apart from four host utilities: ``engine_version``,
``set_engine_log_level``, ``cache_info`` and ``clear_cache``.

.. _arch-pkg-modeling:

``cimba.modeling``: the modeling language
-----------------------------------------

What users write *with*: the ``Model`` base class, the field markers
(``Param``, ``State``, ``Output``, ``Ref``, ``Input``, ``Series``), the entity
classes, the decorators and the verb stubs. On the host these are
declarations and configuration only. Entity objects carry configuration
(``capacity``, ``captured``), decorators tag methods with a role, and verb
stubs raise ``NotInCompiledCode``. ``Model.__setattr__`` validates
assignments at the line where they happen, and ``Sweep`` values live here.

It imports no JIT and no native code. Its meaning inside a trial is given by
the compiler.

.. _arch-pkg-inputs:

``cimba.inputs``: the input-modeling catalogue
----------------------------------------------

Everything about *where variability comes from*, and one of the two parts of
the input-modeling pillar:

* source value types: ``DistributionSource``, ``TraceSource``,
  ``GeneratedRows`` and the ``Source``/``RowSource`` protocols;
* the catalogue: ``dist.*``, ``trace``, ``bootstrap.*``, ``fitted.*``;
* fitting (``fit``), stable stream derivation (``trace_rng``) and custom
  sources (``row_source``);
* provenance: every source's versioned ``describe()``, with SHA-256
  fingerprints of the reference data.

Sources subclass ``modeling.Input``/``Series`` so that assigning one to a
field type-checks. Otherwise the package is pure NumPy, SciPy and
statsmodels, with no JIT and no native code. The compiler never imports it,
so compiled code can't depend on sources.

.. _arch-pkg-schema:

``cimba.schema``: the logical model
-----------------------------------

Two frozen descriptions, built from Python objects by reflection:

``ClassSchema.of(cls)``
   Per class, and cached: every field with its kind, value type, default,
   series step and origin, and optionality, plus processes (with copies and
   priority), hooks, predicates, events and functions (with their annotated
   signatures), gathered through the class hierarchy. ``slots`` fixes the
   dispatch-table order so subclasses stay compatible with their bases.

``Assembly.of(root)``
   Per experiment: a snapshot of the configured tree. It holds every static
   instance with its canonical label, schema and resolved field values
   (entities materialized, lists frozen), plus an object → instance index.
   This is where configuration errors are found and reported as
   ``ModelDefinitionError`` with the instance path.

.. _arch-pkg-layout:

``cimba.layout``: the physical plan
-----------------------------------

Turns logical descriptions into memory:

``RecordLayout.of(schema)``
   The C-compatible record (a NumPy structured dtype) for one class. It
   starts with a pointer to the class's dispatch table, then holds one member
   per field: scalars, a fixed-size ``InputSlot`` per input, an entity handle,
   a pointer per child or reference, and a pointer plus length per list.
   Distinct classes get distinct record types, even when their fields
   coincide.

``TrialImageLayout.of(assembly)``
   Where every instance's record sits inside one trial's memory block, after
   the trial header, and the total block size.

.. _arch-pkg-compiler:

``cimba.compiler``: per-class native code
-----------------------------------------

Compiles each model **class** once and never looks at instances or sources.

* **Views** (``views.py``): Numba record types for model records. Child and
  reference fields become typed pointer loads, ``Ref | None`` becomes an
  optional, and lists become (pointer, length) views with bounds checks.
  Dereferencing a missing reference abandons the trial instead of crashing.
* **Handles** (``handles.py``): Numba overloads that give records their
  methods, such as ``Container.put``, ``Store[T].get``, ``Input.next``,
  ``Series.now`` and ``Resource.acquire``, and implement the verbs. Each maps
  to one external C call. Misuse (a ``Store[Part]`` given a ``Ship``) is a
  typing error at compile time.
* **Functions** (``functions.py``): ``@function`` support. It compiles each
  class's implementations behind a fixed C calling convention derived from
  the annotations, and lowers ``view.f(args)`` to an indirect call through
  the record's dispatch table. Calls are typed under a private key, so a
  function named like an entity method (``get``, ``level``) can't shadow it.
* **Spawning** (``spawning.py``): typing and lowering for
  ``cb.spawn(ModelClass, ...)``, which checks keyword fields against the
  schema at compile time.
* **Classes** (``classes.py``): ``ensure(schema)`` compiles every process,
  hook, predicate and event with ``numba.njit`` over the class's record type,
  wraps them as C callbacks, builds the class dispatch table, and caches the
  result for the process lifetime.

Only ``numba_compat.py`` imports ``numba.core`` internals, which contains
Numba version churn in one file.

.. _arch-pkg-engine:

``cimba.engine``: the native boundary (Python side)
---------------------------------------------------

The one place that loads the native library and touches addresses:

* ``library.py`` loads ``libcimba_py`` with ``ctypes`` and checks the ABI
  version and ``InputSlot`` size at import;
* ``abi.py`` holds NumPy dtypes mirroring the C structs in ``abi.h``;
* ``symbols.py`` registers native symbols with llvmlite so compiled code can
  call them directly;
* ``runtime.py`` provides the host calls: run blocks (with the GIL
  released), seed input slots, detach captures and regenerate distribution
  rows.

.. _arch-pkg-experiments:

``cimba.experiments``: design, binding, running
-----------------------------------------------

The orchestrator, and the only package that sees everything:

* ``design.py`` expands sweeps (in ``Param`` fields, distribution
  parameters, sources and child models) into design points, and derives
  trial seeds with ``SeedSequence``;
* ``run.py`` holds ``Experiment`` and ``Window``. It validates sources
  against the window, plans memory chunks, generates rows for row sources
  once per replication, allocates and binds trial blocks, calls the native
  runner, reruns exhausted trials with extended rows, and assembles
  ``Results``. For swept child models it runs one model tree per option and
  merges the parts.

.. _arch-pkg-results:

``cimba.results``: immutable outcomes
-------------------------------------

Plain, pure data containers: ``Results`` (indexed by model object),
``Samples``, ``InputRecord``, ``Signal`` and ``RunMeta``. Every array is
made read-only, and no native pointer is reachable from a result.

.. _arch-pkg-analysis:

``cimba.analysis``: statistics
------------------------------

``summary``, paired ``compare``, and ``check_input`` for validating sources
against data. It depends only on ``inputs`` and ``results`` (and SciPy and
statsmodels).

.. _arch-pkg-random:

``cimba.random``: model-internal draws
--------------------------------------

Distribution functions usable in compiled code, overloaded onto the engine's
per-trial default random stream. On the host they raise.

.. _arch-pkg-diagrams:

``cimba.diagrams``: structure diagrams
--------------------------------------

Draws configured models without compiling them. It builds a non-strict
``Assembly`` (unbound sources are allowed), then:

* ``structure(model)`` turns the instance tree into a graph of ownership and
  references;
* ``process_graph(model)`` (``processes.py``) parses each process, event and
  hook with ``ast`` and resolves attribute chains against the configured
  objects. It follows children, references, list items, aliases, helper
  functions and the keyword arguments of ``cb.spawn``, which bind a spawned
  class's references and inputs.

Both produce a renderer-neutral ``Graph`` (``graph.py``) that emits Mermaid
or Graphviz DOT. The package depends only on ``modeling`` and ``schema``.

The dependency graph
--------------------

.. code-block:: text

   cimba (facade) ──► modeling, experiments, results, analysis, inputs, random

   experiments ──► compiler, layout, schema, inputs, modeling, results, engine
   compiler    ──► layout, schema, modeling, engine          (never inputs/experiments/results)
   layout      ──► schema, modeling, engine.abi
   schema      ──► modeling, inputs
   inputs      ──► modeling                                  (Input/Series base types only)
   analysis    ──► inputs, results
   diagrams    ──► schema, modeling
   random      ──► engine
   results, engine ──► (nothing in cimba)

Rules enforced by the test suite (``tests/test_architecture.py``):

* ``inputs``, ``modeling``, ``schema``, ``layout``, ``results`` and
  ``analysis`` import no ``numba``, ``llvmlite`` or ``ctypes``. They are pure
  and fast to import and test.
* ``compiler`` imports nothing from ``inputs``, ``experiments`` or
  ``results``, and never an ``Assembly``. Compiled code is a function of the
  class alone.
* Only ``compiler/numba_compat.py`` imports ``numba.core``.
* No module name starts with an underscore.
* Every name in the Windows export list exists in the native library.
