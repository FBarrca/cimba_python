The native runtime
==================

Everything that happens inside a trial happens in one C shared library,
``libcimba_py``. It has no Python C-API dependency, so a single wheel serves
every supported CPython version on a platform. Python reaches it two ways:

* the **host** (``cimba.engine``) calls about ten functions through
  ``ctypes``: run blocks, seed an input slot, release captures, set log
  flags, read versions;
* **compiled model code** calls verbs directly. ``cimba.engine.symbols``
  registers each native symbol's address with llvmlite, and Numba emits
  plain C calls to it. No Python is involved at run time.

.. contents::
   :local:
   :depth: 1

.. _arch-native-engine:

The Cimba engine
----------------

The `Cimba C engine <https://github.com/ambonvik/cimba>`_ is vendored as a
git submodule and linked statically. It provides:

* **stackful coroutines** with hand-written assembly context switches (hence
  NASM) for x86-64 and arm64, so a process can block at any call depth;
* the **event queue** and simulated clock;
* the **entities**: buffers (``Container``), object queues (``Store``),
  priority queues (``PriorityStore``), resource pools (``Resource``),
  conditions and datasets, with their time-weighted statistics;
* the random number generator behind ``cimba.random``;
* **parallel trial execution** (``cimba_run``) over a pool of worker threads,
  with all engine state thread-local;
* **trial abandonment**: an engine error calls ``cimba_trial_abandon``, which
  unwinds the trial and runs a registered cleanup callback, so a failing
  trial never takes the process down.

.. _arch-native-trial:

``trial.c``: the trial runner
-----------------------------

``cpy_run(blocks, count, block_size, workers)`` hands every trial block to
``cimba_run``. For each block, ``run_trial``:

1. reads the **trial descriptor** (shared tables of entities, processes,
   hooks and inputs, each with record offsets) from the block header;
2. initializes the event queue and the engine's random stream from the
   trial seed;
3. creates every entity and writes its handle into the owning record;
4. runs the ``on_start`` hooks, then creates and starts every process copy
   with its callback, record pointer and priority;
5. schedules the window events: start recording at ``warmup``, stop
   recording at ``warmup + duration``, stop processes after the cooldown
   (or starts recording immediately for run-until-idle);
6. executes the event loop until it is empty;
7. runs the ``on_end`` hooks, of static models first, then of spawned
   models still alive;
8. copies the histories of captured entities into malloc'd arrays attached
   to the header, for Python to detach;
9. stops, terminates and destroys processes, spawned models and entities,
   releases input histories, and sets status OK.

The runner is **descriptor-driven**: it contains no model-specific code.
Everything it knows about a model comes from the tables and the callback
addresses the compiler produced.

Abandonment and cleanup
~~~~~~~~~~~~~~~~~~~~~~~

Every trial registers ``abandoned_trial`` as its cleanup callback. If model
code triggers an engine error (a null reference, a bad index, an input that
fails or is exhausted, an invalid schedule), the engine unwinds and calls it.
The callback:

* inspects the input slots to classify the outcome: *exhausted* (rerun with
  a longer row), *ended*, or *failed*, and writes a reason such as
  ``"network.facilities[2].demand exhausted after 359 values"``;
* frees spawned models, input histories, captures, process and entity
  tables, and the random stream.

.. mermaid::

   stateDiagram-v2
     direction LR
     [*] --> PENDING
     PENDING --> RUNNING: cpy_run
     RUNNING --> OK: event loop done, on_end ran
     RUNNING --> FAILED: engine error or input "fail"
     RUNNING --> EXHAUSTED: input "extend" ran out
     RUNNING --> ENDED: input "end_trial" ran out
     EXHAUSTED --> RUNNING: host regenerates rows 2×, reruns
     OK --> [*]
     FAILED --> [*]
     ENDED --> [*]

Every native object is created and destroyed exactly once per trial, even on
abandonment, so a worker thread can run thousands of trials, some of them
failing, without leaking.

Spawned models
~~~~~~~~~~~~~~

``cpy_model_allocate`` gives a spawned model a zeroed record from the heap,
tracked in a per-trial list and stamped with its class's dispatch table.
``cpy_model_input_bind`` re-seeds cloned input slots.
``cpy_model_start`` runs the new model's ``on_start`` hooks and schedules its
processes at the current time. ``cpy_model_release`` stops its processes and
marks it released. All spawned records are freed when the trial ends, so a
reference to a released model never dangles within the trial.

.. _arch-native-inputs:

``inputs.c``: the input runtime
-------------------------------

Serves ``next()``, ``now()`` and ``at(t)`` for every input slot. It contains
the per-slot random generator and the implementations of the 27
distributions, row replay with bounds checks, series bucketing with
remembered history, and the exhaustion policies. See
:doc:`inputs` for the semantics.

.. _arch-native-verbs:

``verbs.c`` and ``nbshim.c``: what model code calls
---------------------------------------------------

Thin C functions that give compiled code a stable, typed entry point into
the engine:

* typed store operations: ``int`` and ``float`` payloads are bit-packed into
  the engine's pointer-sized items, and model payloads are record pointers;
* priority-store tickets (``enqueue``, ``position``, ``cancel``);
* conditions whose demand function calls a compiled predicate through the
  class dispatch table;
* scheduled events that call a compiled event method with its record;
* process handles, timers, interrupts and status;
* buffer, resource and dataset statistics; the ``cimba.random`` draws;
  logging with user flags; ``cpy_end_trial``.

The ABI
-------

``abi.h`` defines the structures shared by Python and C: the trial header,
the input slot and the descriptor tables. ``cimba.engine.abi`` mirrors them
as NumPy dtypes, and at import time Python checks the ABI version and the
input slot size against the library. A mismatch refuses to load rather than
corrupting memory.

On Windows, exported symbols are listed in ``cimba.def``. A test checks that
every listed name exists in the built library.

Building
--------

``meson.build`` compiles the engine submodule as a static library and links
it with ``trial.c``, ``inputs.c``, ``verbs.c`` and ``nbshim.c`` into
``libcimba_py``. Wheels are built with ``cibuildwheel`` for Linux x86_64,
Windows AMD64 (clang-cl) and macOS arm64, with the library included in the
package directory where ``cimba.engine.library`` finds it.
