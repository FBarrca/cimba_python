The input-modeling pillar
=========================

Input modeling, meaning where the variability in a simulation comes from, is
one of the two pillars of Cimba Python, alongside the modeling language. It
spans both sides of the native boundary: ``cimba.inputs`` describes sources
in Python, and ``inputs.c`` serves their values to running trials.

.. raw:: html
   :file: ../static/diagrams/input_pillar.svg.html

The design goals
----------------

1. **Source-agnostic model code.** Model code says *what* it consumes
   (``self.demand.now()``), never how the values are produced. Changing the
   source changes values only, never compiled code.
2. **Data as a first-class citizen.** Recorded traces and resamples are as
   easy to use as a textbook distribution, and are checked, fingerprinted and
   recorded in the results.
3. **Synchronized randomness.** Each input has its own stream, so common
   random numbers stay aligned per input, and one input's consumption never
   shifts another's values.
4. **No sizing by hand.** Users never choose row lengths; running out is
   handled by explicit, deterministic policies.
5. **No Python in the hot loop.** Reading an input in a trial is one C
   call, whatever the source.

Where values come from
----------------------

.. list-table::
   :header-rows: 1
   :widths: 16 34 25 25

   * - Source kind
     - Values produced
     - Random stream
     - Finite?
   * - distribution
     - In the trial, by ``inputs.c``, one per read
     - Per input slot, seeded from trial seed ⊕ hash(path)
     - Never
   * - trace
     - The recorded array, shared read-only by all trials
     - None
     - Yes: ``fail``, ``wrap`` or ``end_trial``
   * - resample, fitted
     - On the host, by the source's generator, one row per replication
     - ``trace_rng(trial seed, tag or path)``
     - Rows are extended on demand (``extend``)

Row sources are generated **per replication, not per trial**. Under common
seeding, replication *r* has the same seed at every design point, so its rows
are generated once and shared by all points that bind the same source. Memory
and generation time scale with replications, not with design points × replications.
If the rows don't fit in ``input_memory``, replications are processed in
chunks.

Why not resample inside the trial? Block, joint and model-based resampling
need structure (blocks, shared indices, fitted AR models) that is natural in
NumPy and awkward in compiled trial code. Generating rows outside the trial
and replaying them inside keeps both sides simple, and the results are exact
and reproducible either way. Distributions are the one kind of source drawn
inside the trial: they're cheap, infinite and need no data.

The input slot
--------------

Every ``Input``/``Series`` field occupies a fixed-size ``InputSlot`` in its
model's record:

.. code-block:: c

   typedef struct cpy_input_slot {
       uint32_t kind;             /* 1 distribution, 2 rows                    */
       uint32_t policy;           /* fail, wrap, end_trial, extend             */
       uint32_t distribution;     /* which distribution, for kind 1            */
       uint32_t status;           /* ok, failed, ended, exhausted              */
       double   parameters[4];    /* distribution parameters                   */
       uint64_t stream_state[4];  /* this input's own random stream            */
       const double *data;        /* row / trace / categorical tables          */
       uint64_t length, cursor;   /* row length and read position              */
       double   step, origin;     /* Series bucket geometry                    */
       int64_t  last_bucket;
       double   last_value;
       double  *history;          /* remembered buckets of distribution series */
       uint64_t history_capacity;
   } cpy_input_slot;

Compiled model code holds a handle to its slot. ``next()``, ``now()`` and
``at(t)`` each compile to one call (``cpy_input_next``, ``cpy_series_now``,
``cpy_series_at``) that dispatches on ``kind``. The compiled code is
identical whatever source is bound. After the call, the handle checks the
slot's status. Anything other than OK abandons the trial, and the runner
reads the slot to decide what happened.

Streams
-------

A distribution slot's ``stream_state`` is seeded on the host, when the block
is bound, from the trial seed XOR a hash of the input's canonical path. The
generator is a small, fast 256-bit-state PRNG private to ``inputs.c``, so
input streams never interact with the engine's default stream (used by
``cimba.random``) or with each other.

Row sources are seeded by ``trace_rng(trial_seed, tag)``, where the tag is
the source's own tag if it has one, otherwise the input's path. Members of a
``bootstrap.joint`` share a tag, so they receive identical generators and
pick the same blocks. That is how cross-series structure survives
resampling.

Spawned models receive a *copy* of their spawner's slot.
``cpy_model_input_bind`` (in ``trial.c``) re-seeds it from the trial seed and a
per-trial spawn counter, and resets its cursor and history. A spawned
distribution input gets its own deterministic stream, independent of worker
scheduling. A spawned row input replays the same row from its start.

Series buckets
--------------

A series maps time to a bucket, ``floor((t - origin) / step)``:

* for **rows**, bucket *k* is ``data[k]``, and the cursor records the
  highest bucket read plus one (reported as ``consumed``);
* for **distributions**, buckets are drawn in order and remembered in the
  slot's ``history``. Reading bucket 10 first draws buckets 0–10, so bucket
  *k* has the same value however the model accesses it, and ``at()`` can look
  back.

When the window is finite, the host computes the number of buckets it
needs, ``ceil((warmup + duration + cooldown − origin) / step)``, and
generates at least that many rows. A ``"fail"`` trace that is too short is
rejected when the experiment is created.

Exhaustion and extension
------------------------

When a row slot runs past its end, its policy decides:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Policy
     - In ``inputs.c``
   * - ``wrap``
     - Continue from ``data[cursor % length]``.
   * - ``fail``
     - Set status *failed*. The trial is abandoned and the reason names the
       input and the number of values consumed.
   * - ``end_trial``
     - Set status *ended*. The trial is abandoned with the reason recorded;
       in 0.7.0 it is reported as failed.
   * - ``extend``
     - Set status *exhausted*. The trial is abandoned with status
       ``EXHAUSTED``, to be rerun.

For ``extend``, after the native call returns, ``experiments``:

1. finds the exhausted slot (or, for a spawned model's clone, every row
   source bound in that design point);
2. regenerates the row at **twice** the length from the **same** generator;
3. verifies that the new row starts with the old one, which is prefix
   stability, and raises if a source breaks that promise;
4. restores the trial's block from a pristine copy, points the slot at the
   longer row, and reruns just that trial;
5. gives up after six doublings, or at the source's ``max_length``, and
   records a failure reason.

Because the rerun sees exactly the same values up to the old end, the
extended trial's result is **identical** to the result it would have had
with a long enough row from the start. Extension changes cost, never
results.

Provenance
----------

Every source has a versioned ``describe()``: its method, its parameters and,
for data-based sources, a SHA-256 fingerprint of the reference data (and of
the whole panel for joint members). Reference arrays are copied and made
read-only when a source is built, and row sources re-verify the fingerprint
before generating.

The results carry, per input, one ``describe()`` per design point, the
per-trial ``consumed`` and ``extended`` counts, and ``rows(p, r)``, which
regenerates the exact values from the recorded seed and source. Nothing
needs to be stored to replay any trial's inputs.
