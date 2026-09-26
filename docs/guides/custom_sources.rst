Write a custom input source
===========================

**Goal:** feed an input from your own generator, such as a regime-switching
resample or a simulator of your own, with the same guarantees as the
built-in sources.

The row-source contract
-----------------------

Resampling and fitted sources are **row sources**. On the host, before the
trials run, Cimba asks each one for a row of values per replication, then
replays the rows inside the trials. Build one with
:func:`cimba.inputs.row_source`:

.. code-block:: python

   inputs.row_source(
       method,                  # a name for provenance, e.g. "custom.regime_resample"
       data,                    # the reference data (copied and fingerprinted)
       draw,                    # draw(rng, length) -> 1-D float array of `length`
       parameters={...},        # recorded in describe()
       tag=None,                # share a tag to share a random stream (joint sources)
       prefix_stable=True,      # see below
       max_length=None,         # a hard upper bound on row length, if any
       length_hint=None,        # a sensible first length (default 1,024)
       on_exhausted="extend",   # or "fail", "wrap", "end_trial"
   )

``draw`` receives a ``numpy.random.Generator`` that Cimba derives from the
replication's seed and the input's identity. Use **only** that generator, so
that results are reproducible and synchronized across design points.

An example: regime-switching demand
-----------------------------------

Suppose demand alternates between busy and quiet spells, and you want
resamples that keep that pattern:

.. code-block:: python

   import numpy as np
   from cimba import inputs

   median = np.median(history)
   busy_days = history[history > median]
   quiet_days = history[history <= median]
   SWITCH = 0.05                       # daily probability of changing regime

   def draw(rng, length):
       out = np.empty(length)
       busy = rng.random() < 0.5
       for i in range(length):
           if rng.random() < SWITCH:
               busy = not busy
           pool = busy_days if busy else quiet_days
           out[i] = pool[rng.integers(len(pool))]
       return out

   regimes = inputs.row_source("custom.regime_resample", history, draw,
                               parameters={"switch": SWITCH})
   shop.demand = regimes

Check it like any other source:

.. code-block:: python

   report = cb.analysis.check_input(regimes, reference=history)
   report.generated_acf[:3]     # strong persistence, by construction

Prefix stability
----------------

A source is **prefix-stable** if ``draw(rng, 2n)`` starts with exactly the
values of ``draw(rng, n)`` for the same generator state. Cimba relies on it
to extend rows transparently: when a trial needs more values, it regenerates
a longer row and reruns the trial, and the rerun must see the same history.

The rule of thumb: **draw in time order, and never let the length change
what you draw early on.** The example above is prefix-stable because each
step consumes the same random numbers whatever ``length`` is. These break it:

* ``rng.permutation(length)`` or ``rng.choice(n, size=length)`` followed
  by length-dependent processing, such as normalizing by the row's mean;
* drawing a length-sized block of one kind of number before the loop.
  ``rng.normal(size=length)`` then ``rng.integers(..., size=length)`` changes
  the integers when the length changes.

Test it directly:

.. code-block:: python

   short = regimes.generate([inputs.trace_rng(42, "test")], 100)[0]
   long = regimes.generate([inputs.trace_rng(42, "test")], 200)[0]
   assert np.array_equal(short, long[:100])

If your generator can't be prefix-stable, declare it with
``prefix_stable=False`` and a policy other than ``"extend"``, and give a
``length_hint`` long enough for your window. Cimba then fails a trial
that runs out instead of extending it.

Joint custom sources
--------------------

Sources with the same ``tag`` receive identically seeded generators in each
replication. To keep several custom inputs synchronized, as
``bootstrap.joint`` does, build them with the same tag and have their
``draw`` functions consume the generator identically.
