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
  priority queues (``PriorityStore``), conditions and datasets, with their
  time-weighted statistics, and the waiting-room and holding machinery that
  ``Resource`` builds on (see :ref:`arch-native-resource`);
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
3. creates every entity (putting a container's ``initial`` units in) and
   writes its handle into the owning record, then subscribes each observing
   condition to the entities it observes (see :ref:`arch-native-observers`);
4. runs the ``on_start`` hooks, then creates and starts every process copy
   with its callback, record pointer and priority;
5. schedules the window events: start recording at ``warmup``, stop
   recording at ``warmup + duration``, stop every process, static and
   spawned, after the cooldown
   (or starts recording immediately for run-until-idle);
6. executes the event loop until it is empty;
7. runs the ``on_end`` hooks, of static models first, then of spawned
   models still alive;
8. copies the histories of captured entities into malloc'd arrays attached
   to the header, for Python to detach;
9. stops, terminates and destroys processes and spawned models,
   unsubscribes the observing conditions, destroys the entities, releases
   input histories, and sets status OK.

The runner is **descriptor-driven**: it contains no model-specific code.
Everything it knows about a model comes from the tables and the callback
addresses the compiler produced.

Abandonment and cleanup
~~~~~~~~~~~~~~~~~~~~~~~

Every trial registers ``abandoned_trial`` as its cleanup callback. If model
code triggers an engine error (a null reference, a bad index, an input that
fails or is exhausted, an invalid schedule), the engine unwinds and calls it.
The callback:

* unsubscribes the observing conditions while the entities still exist
  (the engine tears them down after the callback);
* inspects the input slots to classify the outcome: *exhausted* (rerun with
  a longer row) or *failed*, and writes a reason such as
  ``"network.facilities[2].demand exhausted after 359 values"``;
* frees spawned models, input histories, captures, process and entity
  tables, and the random stream.

.. mermaid::

   stateDiagram-v2
     direction LR
     [*] --> PENDING
     PENDING --> RUNNING: cpy_run
     RUNNING --> OK: event loop done (or end_trial), on_end ran
     RUNNING --> FAILED: engine error or input "fail"
     RUNNING --> EXHAUSTED: input "extend" ran out
     EXHAUSTED --> RUNNING: host regenerates rows 2×, reruns
     OK --> [*]
     FAILED --> [*]

Every native object is created and destroyed exactly once per trial, even on
abandonment, so a worker thread can run thousands of trials, some of them
failing, without leaking.

.. _arch-native-observers:

Observing conditions
~~~~~~~~~~~~~~~~~~~~

The descriptor's observation table lists (condition, observed entity)
pairs, as indices into the entity table, built from
``Condition(observe=...)`` and checked against the model tree on the host.
The engine's mechanism for this is the observer list of a resource guard:
``cmb_resourceguard_register(subject, observer)`` forwards every signal of
the subject guard to the observer as one ``cmb_resourceguard_signal``. That
call looks at the observer's front waiter only, so a condition whose waiters
have different predicates would leave a true one behind a false one. The
runner therefore registers a **relay** guard per observation: its queue holds
one permanent entry whose demand function calls ``cmb_condition_signal`` (which
evaluates every waiter, exactly as ``Condition.signal()``) and declines the
grant. A relay is registered on the resource's guard, or on both the front
and rear guards of a store or container, so puts and gets both signal. It
uses only the engine's public guard API and is unregistered and terminated
before the entities are destroyed, on both the normal and the abandonment
path, so observer tags never outlive a trial.

Spawned models
~~~~~~~~~~~~~~

``cpy_model_allocate`` gives a spawned model a zeroed record from the heap,
tracked in a per-trial list and stamped with its class's dispatch table.
``cpy_model_input_bind`` re-seeds cloned input slots.
``cpy_model_start`` runs the new model's ``on_start`` hooks and schedules its
processes at the current time. ``cpy_model_release`` stops its processes and
marks it released. All spawned records are freed when the trial ends, so a
reference to a released model never dangles within the trial.

.. _arch-native-arrays:

Array storage
~~~~~~~~~~~~~

A ``cb.Array`` field is two words in the record: the address of its element
buffer and its length. The record stays fixed-size per class; where the
elements live depends on the instance:

* **Static models.** The trial image reserves the elements after the
  records (``TrialImageLayout``), sized for the longest option of a swept
  length. The host writes each trial's address, length and initial values
  into the block, so every trial starts from the configured contents, an
  input-extension rerun restores them with the block, and nothing is
  allocated or freed natively.
* **Spawned models.** ``cpy_model_array(record, offset, length, size,
  initial, label)``, called by the spawn right after ``cpy_model_allocate``,
  allocates a zeroed buffer (or copies the class default's values), writes
  its address and length into the record and links it to the model's node.
  A negative length fails the trial. ``cpy_model_release`` frees the
  model's buffers at once and sets their length to -1, since nothing of a
  released model runs again. The node teardown that frees records at trial
  end, on the normal and the abandonment path, frees any buffer still
  linked. Each buffer is therefore freed exactly once per trial.

Compiled code holds an array as the address of its two words, not as a copy
of them, so every access reads the current buffer and length: a view kept
across ``release`` sees length -1 and can't reach freed memory. Each index is
compared with the length as a signed value. A miss calls
``cpy_array_index_fail``, which fails the trial with the field, index and
length, or says that the model was released.

References are pointers in the record, 0 for ``None``. Assigning one in
compiled code stores the new pointer; a required reference is never 0 once
the model is bound or spawned, and an optional value is checked before it is
stored there (``cpy_reference_fail``).

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

.. _arch-native-resource:

``resource.c``: Resource
------------------------

``Resource`` is implemented here rather than in the engine, so that request
priorities, first come, first served across partial waits and a changing
capacity need no change to the vendored engine. The engine stays at its
pinned commit, unmodified. The acquire, preempt and release algorithms, the
holders queue and the histories follow the engine's ``cmb_resourcepool``, so
models get the same results as on it, except that a partially served request
keeps its place in line.

It uses these engine extension points, all present in the unmodified engine:

* **The waiting room is a** ``cmb_resourceguard``. Instead of calling
  ``cmb_resourceguard_wait``, the resource enqueues into the guard's hashheap
  itself, with the request's own rank (explicit or the process priority),
  its first entry time, and a key whose middle bits are the request's
  sequence number. It registers the ``RESOURCE`` awaitable the engine
  expects and suspends. From then on the engine treats the entry as any
  guarded wait: ``cmb_resourceguard_signal`` grants the front request with a
  wakeup at the process priority, an interrupt or a stop removes it, and
  ``priority_set`` re-ranks it. Every wait gets a fresh key, with the top
  bit set so it never collides with the engine's own keys, so a stale grant
  from an earlier wait is discarded by the engine's key check. The guard's
  observer list is the hook for signalling a ``Condition``.
* **The holdable interface** (``cmi_holdable``). A process carries the
  resource's holdable tag on its resource list while it holds units *or*
  waits there. The engine calls ``drop`` when a holder exits or is stopped,
  and ``reprio`` from ``priority_set`` after it has re-ranked the waiting
  entry: ``reprio`` re-ranks a holding that tracks the process priority and
  puts an explicitly ranked waiting entry back at its own rank.
* **Process interrupts** for preemption: a victim loses its holding, its
  waits are cancelled (``cmi_process_cancel_awaiteds``), and it gets
  ``CMB_PROCESS_PREEMPTED`` via ``cmb_process_interrupt`` at its own
  priority.
* **The memory registry**, as for ``FractionalContainer``, so an abandoned
  trial tears the resource down like the engine's own objects.

Every release, grant and capacity change signals the guard, so a
``Condition`` observing the resource (see :ref:`arch-native-observers`)
sees each change through the guard's observer list.

**Acquire any.** ``cpy_resource_acquire_any(resources, count, ranked,
rank)`` takes one unit from whichever member grants first, without engine
changes:

* If a member has room, it takes a unit from the first such member at once.
* Otherwise it enqueues one entry into *every* member's guard, all with the
  same rank, entry time and key (a fresh key from the same sequence as
  ``acquire``, so ``acquire`` and ``acquire_any`` requests are ordered
  first come, first served together), registers one ``RESOURCE`` awaitable
  per member with that key, and suspends. A listed duplicate gets no second
  entry.
* The entries' demand function is the **grant hook**. The engine calls it
  from ``cmb_resourceguard_signal`` on the front entry and, when it returns
  true, dequeues the entry and schedules the wakeup. Before returning true,
  the hook records the granting member and removes the request's entries and
  awaitables from every other member. From then on the engine sees one
  ordinary granted guarded wait, so a second member can never grant it, and
  the engine's key check discards the wakeup if the grant goes stale.
* Interrupts, stops and preemption remove the remaining entries through the
  engine's ``cmi_process_cancel_awaiteds``, as for any guarded wait. On
  resumption the call cleans up any entry still queued (a timer resumes it
  directly), and on a signal other than success or cancellation it signals
  every member, passing on a grant that may have been pending.
* ``priority_set`` re-ranks every member's entry through the engine, and
  each member's ``reprio`` puts an explicitly ranked entry back, because the
  entry's context starts with the same ``struct request`` as an ``acquire``.
* On success it takes the unit as an ordinary holding of the granting member,
  ranked by the request. If another process took the unit between the grant
  and the wakeup, it looks again and waits again at the same place.

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
it with ``trial.c``, ``inputs.c``, ``verbs.c``, ``nbshim.c``,
``fractional.c`` and ``resource.c`` into
``libcimba_py``. Wheels are built with ``cibuildwheel`` for Linux x86_64 and
aarch64, Windows AMD64 (clang-cl) and macOS arm64, with the library included in the
package directory where ``cimba.engine.library`` finds it.
