Inputs and sources
==================

This page covers the idea that sets Cimba Python apart:

   **A model declares what variability it consumes. Where the values come
   from is configuration.**

The same compiled model can draw service times from a gamma distribution
today, replay last year's log tomorrow, and run on a thousand bootstrap
resamples of that log next week. Its code doesn't change, and neither do the
other inputs' values.

Why inputs are separate
-----------------------

In most simulation libraries, randomness is written into the model:
``service = rng.gamma(2.0, 0.5)``. That's fine until you want to use data.
Replaying a log or resampling it then means rewriting every place the
distribution was drawn, adding cursors and array bounds, and worrying about
running out. It also ties every random draw to one shared stream, so drawing
one extra number anywhere shifts every value after it. Comparisons between
scenarios get noisier.

Cimba splits the concern in two:

.. list-table::
   :widths: 25 75

   * - **Input** (in the model)
     - *What* is consumed: "a sequence of inter-arrival times", "daily demand
       for this store". Declared as a field, read with ``next()`` or
       ``now()``.
   * - **Source** (in the configuration)
     - *How* values are produced: a distribution, a recording, a resample or
       a fitted time-series model. Assigned to the input on the model
       object, or swept.

Declaring inputs
----------------

.. code-block:: python

   class Store(cb.Model):
       # A sequence: each call to next() consumes one value.
       lead_time: cb.Input[float] = inputs.dist.gamma(shape=2.0, scale=1.5)

       # A time series: one value per bucket of `step` time units.
       demand: cb.Series[float] = cb.Series(step=1.0, origin=0.0)

       # Integer and boolean inputs are converted when read.
       batch: cb.Input[int] = inputs.dist.poisson(mean=3)
       open_today: cb.Series[bool] = cb.Series(step=1.0)

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Kind
     - Read in model code
     - Semantics
   * - ``Input[T]``
     - ``x.next()``
     - Returns the next value of the sequence and advances the cursor.
   * -
     - ``x.remaining()``
     - Values left before the end of a finite source; ``-1`` for a
       distribution (never runs out).
   * - ``Series[T]``
     - ``x.now()``
     - The value of the bucket containing ``cb.now()``.
   * -
     - ``x.at(t)``
     - The value of the bucket containing time ``t``.

A ``Series`` maps time to a **bucket**, ``floor((t - origin) / step)``, and
bucket *k* is the *k*-th value of the source. Reading the same bucket twice
returns the same value. Skipping buckets doesn't shift later ones. A
distribution-fed series draws buckets in order and remembers them, so
``at()`` can look back at any earlier bucket. Reading before ``origin`` is an
error.

The default after ``=`` is optional for ``Input`` (a class default source)
and required for ``Series`` (its ``step``/``origin``). A ``Series`` declared
with ``cb.Series(step=...)`` still needs a source bound before running.

The four kinds of source
------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 32 26 24

   * - Kind
     - Constructed with
     - Values produced
     - Finite?
   * - distribution
     - ``inputs.dist.exponential(mean=2)`` … (27 distributions)
     - inside the trial, on the input's own stream
     - never
   * - trace
     - ``inputs.trace(array, on_exhausted=...)``
     - the recorded array, replayed as-is
     - yes
   * - resample
     - ``inputs.bootstrap.iid / moving_block / circular_block /
       stationary / joint / intermittent``
     - rows generated on the host per replication
     - regenerated as needed
   * - fitted model
     - ``inputs.fitted.residual / wild / sieve / intermittent``
     - rows generated on the host per replication
     - regenerated as needed (``wild``: up to its history length)

Every source is an immutable value with a ``describe()`` method that returns
its provenance: method, parameters and, for data-based sources, a SHA-256
fingerprint of the reference data. Reference arrays are copied when the
source is built, so changing your array afterwards can't silently change a
running study.

The full catalogue, with every parameter, is in
:doc:`../reference/inputs`.

Binding a source
----------------

A source reaches an input in one of three ways, most specific first:

.. code-block:: python

   store = Store()

   # 1. assign on the instance (in __init__ or later)
   store.lead_time = inputs.bootstrap.iid(observed_lead_times)

   # 2. sweep several sources: one design point each
   store.lead_time = cb.sweep(inputs.dist.gamma(shape=2.0, scale=1.5),
                              inputs.bootstrap.iid(observed_lead_times))

   # 3. otherwise the class default applies

Distribution parameters can also be swept:
``inputs.dist.exponential(mean=cb.sweep(1.0, 1.5, 2.0))``, or derived from
another sweep with ``axis.map(f)``.

Binding never touches compiled code. A class compiled once serves every
source you bind to it.

Random streams and common random numbers
----------------------------------------

Each input owns an **independent random stream** in each trial. It is derived
from the replication's seed and the input's identity: its canonical path
(``network.facilities[2].demand``) or an explicit source ``tag``. This
has two important consequences:

* **Inputs don't disturb each other.** If one process draws twice as many
  service times, the arrival times don't change. You can add, remove or
  reorder consumption in one part of the model without reshuffling the rest.
* **Design points stay synchronized.** Under the default *common* seeding,
  replication *r* has the same seed at every design point. So an input's
  values in replication *r* are the same across design points (as long as its
  source is the same). Comparisons between points are then *paired*, and
  much sharper. See :doc:`experiments`.

Joint members share a tag, so the members of
``inputs.bootstrap.joint(panel, ...)`` draw the same blocks of rows.
That's how cross-series structure is preserved.

.. note::

   Renaming or moving an untagged input changes its random stream. Use a
   stable tag, such as ``inputs.dist.normal(sd=2, tag="vendor:7:lateness")``,
   to preserve draws across refactors. Use distinct tags for independent
   streams; matching tags and distribution parameters reproduce the same
   draws for a given trial seed.

Model-internal randomness that isn't a real-world input, such as routing coin
flips, can use :mod:`cimba.random` instead. It draws from the trial's
default stream.

When a finite source runs out
-----------------------------

A trace has a fixed length; a resample is generated at some length. What
happens when model code asks for more is set per source with
``on_exhausted``:

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Policy
     - Default for
     - Behavior
   * - ``"fail"``
     - ``inputs.trace``, ``fitted.wild``
     - The trial is abandoned and recorded as failed, with a reason such as
       ``"shop.gaps exhausted after 1000 values"``. Other trials continue.
   * - ``"wrap"``
     - (opt in)
     - Start again from the first value.
   * - ``"end_trial"``
     - (opt in)
     - End the trial when the input runs out, exactly like ``cb.end_trial()``:
       the event loop stops, ``on_end`` hooks run and the trial succeeds.
       ``consumed`` tells you how far it got.
   * - ``"extend"``
     - bootstrap and fitted sources
     - Transparently regenerate a longer row and rerun the trial (below).
       Only for prefix-stable sources.

``inputs.trace(data, on_exhausted=...)`` accepts ``"fail"``, ``"wrap"`` and
``"end_trial"``. The built-in resampling and fitted sources choose their own
policy; custom row sources built with :func:`cimba.inputs.row_source` accept
any of the four. A ``Series`` bound to a ``"fail"`` trace that is too short for
the window is rejected as soon as you create the ``Experiment``.

Transparent extension
~~~~~~~~~~~~~~~~~~~~~

Resampling and fitted sources are **prefix-stable**: generating *2n* values
from the same random stream yields the same first *n* values as generating
*n*. Cimba uses this to make row length a non-issue:

1. Each replication's row starts at a length hint. For a ``Series`` that's
   enough buckets to cover the whole window; otherwise it's the source's
   ``length_hint``, or 1,024 values.
2. If a trial reads past the end, it stops with an *exhausted* status.
3. After the batch, Cimba regenerates that row at double the length and
   reruns only that trial, up to six times.
4. Because of prefix stability, the rerun sees exactly the values it saw
   before, plus more. The result is identical to having generated enough rows
   up front.

``results[model].x.extended`` counts extensions per trial;
``results.meta.extensions`` gives the total. Sources with a hard limit
(``fitted.wild`` can't go beyond its history) fail with a clear reason
instead of extending past it.

Where rows are generated
------------------------

Distribution sources draw inside the trial, in C, one value per ``next()``.
Resample and fitted sources are generated **on the host, once per
replication**, vectorized, and shared by every design point that uses the same
source. They are replayed in the trial from read-only memory, which keeps
Python out of the hot loop and memory proportional to replications, not
trials. If the rows for all replications would exceed ``input_memory``
(default 1 GiB, an argument of ``run()``), replications are processed in
chunks. Results are identical either way.

Spawned models and inputs
-------------------------

A dynamic model with an input field is spawned with an existing input of the
spawner: ``cb.spawn(Order, lead_time=self.lead_time)``. The new model gets its
own cursor and its own stream, derived from the trial seed and the spawn
order. Two spawned orders don't share values, and runs stay reproducible
regardless of worker count.

After the run: provenance, consumption, replay
----------------------------------------------

Every input appears in the results as an :class:`InputRecord`:

.. code-block:: python

   record = results[store].lead_time
   record.source       # tuple of describe() dicts, one per design point
   record.consumed     # (points, replications): values read in each trial
   record.extended     # (points, replications): extensions in each trial
   record.rows(p, r)   # the exact values trial (p, r) consumed

``rows`` regenerates the values on demand. They are never stored, so keeping
results around stays cheap.

The input-modeling workflow
---------------------------

Putting it together, a data-driven study follows these steps:

.. code-block:: text

   look      at the data: continuous? discrete? trend? seasonality? several related series?
   choose    candidate sources: inputs.fit(data) for distributions; bootstrap for shape;
             fitted.* for trend/season; bootstrap.joint for panels
   check     cb.analysis.check_input(source, reference=data): moments, quantiles, ACF, PACF, KS
   bind      model.x = source   (or cb.sweep(source_a, source_b) to compare)
   run       results = cb.Experiment(...).run()
   inspect   results[model].x.source / consumed / extended / rows(p, r)
   compare   cb.analysis.compare(results[model].output, a=0, b=1)

:doc:`../guides/input_models` walks through it with code.
