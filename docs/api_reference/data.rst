Datasets, Summaries and Reporting
=================================

``sim.Dataset`` is a trial-local list of untimed numeric observations. Add one
observation with ``env.waits.add(value)`` each time the measured event happens.
At collection time, call methods such as ``env.waits.mean()`` or
``env.waits.quantile(0.95)`` to turn that list into the metric you want to
report for the current replication. Time-weighted histories are attached to
simulation entities automatically and read back with ``env.<entity>.history()``,
which returns a time-series handle for the same method-style summary calls.

Datasets
--------

Declare datasets as model or component fields:

.. code-block:: python

   class Clinic(sim.Model):
       waits: sim.Dataset
       avg_wait: sim.Output
       p95_wait: sim.Output

       @sim.collect
       def collect_stats(env: "Clinic"):
           env.avg_wait = env.waits.mean()
           env.p95_wait = env.waits.quantile(0.95)

Inside process code, each ``add()`` appends one sample to the current trial's
dataset:

.. code-block:: python

   env.waits.add(wait_time)

At collection time, summarize that trial-local list into the outputs you care
about:

.. code-block:: python

   class Clinic(sim.Model):
       waits: sim.Dataset
       avg_wait: sim.Output
       p95_wait: sim.Output

       @sim.collect
       def collect_stats(env: "Clinic"):
           env.avg_wait = env.waits.mean()
           env.p95_wait = env.waits.quantile(0.95)

A dataset field is created separately for every trial row. If an experiment
uses ``replications=50``, then ``env.waits`` is 50 independent datasets, not
one dataset shared by all replications. Each replication records its own
samples. A model-level or component-level
``@sim.collect`` callback, writes one output value per replication for each
metric you choose. For example, ``exp["avg_wait"]`` contains 50
per-replication averages, while ``exp["p95_wait"]`` contains 50
per-replication percentile estimates. Use ``exp.summary()`` or ordinary Python
after ``exp.run()`` to summarize those output values across replications.

Datasets are reset when the measurement window opens after warmup. Values
added before warmup are discarded, so collectors see the samples for the
measured part of that trial.

Datasets support method-style compiled calls: ``add()``, ``count()``,
``mean()``, ``min()``, ``max()``, ``std()``, ``median()``, ``quantile()``,
``capture()``, ``print()``, ``print_file()``, ``fivenum()``, ``fivenum_file()``,
``histogram()``, ``histogram_file()``, ``correlogram()``,
``correlogram_file()``, ``pacf_correlogram()``, and
``pacf_correlogram_file()``.

Use ``capture()`` when you need the raw per-replication samples in Python
without writing a file. Declare it in the model-level collector next to the
ordinary scalar outputs:

.. code-block:: python

   class Clinic(sim.Model):
       waits: sim.Dataset
       avg_wait: sim.Output

       @sim.collect
       def collect_stats(env: "Clinic"):
           env.avg_wait = env.waits.mean()
           env.waits.capture()

After ``exp.run()``, read the captured sample arrays from the experiment:

.. code-block:: python

   waits0 = exp.dataset("waits", trial=0)
   waits_by_trial = exp.datasets("waits")

Each returned dataset array is one-dimensional ``float64``. Captured names use
the flattened field path, such as ``"waits"`` or ``"station__waits"``. The
arrays are trial-local and aligned with the experiment row order. Component
fields can be captured from the model-level collector with paths such as
``env.station.waits.capture()``.

Time series
-----------

Unlike datasets, a time series is not something you populate yourself. Every
stateful native entity -- ``sim.Queue``, ``sim.Resource``, ``sim.Pool``,
``sim.Store``, and each priority queue in ``sim.PQueues`` -- has its value
(queue length, resource holders, pool holders, store length) recorded by the
engine automatically for as long as the trial runs. Calling ``.history()`` on the entity itself hands back that recording
as a time-series handle:

.. code-block:: python

   class Clinic(sim.Model):
       waiting_room: sim.Queue
       nurse: sim.Resource
       mean_queue_len: sim.Output

       @sim.collect
       def collect_stats(env: "Clinic"):
           env.mean_queue_len = env.waiting_room.history().mean()

``.history()`` is available on every field kind that records one:
``sim.Queue``, ``sim.Resource``, ``sim.Pool``, ``sim.Store``, and indexed
elements of ``sim.PQueues`` (``env.pqs[i].history()``). It works the same way
inside component methods (``self.waiting_room.history().mean()``).

The handle returned by ``.history()`` summarizes and reports
**time-weighted** statistics: a value the queue held for a long stretch
counts more than one held only briefly, which is why these are separate
``.history()...`` methods rather than reusing the dataset methods above
(whose statistics are unweighted, one sample = one vote). The supported
chained methods are:

``count()``, ``min()``, ``max()``, ``mean()``, ``std()``, ``median()``,
``capture()``, ``print()``, ``print_file()``, ``fivenum()``, ``fivenum_file()``,
``histogram()``, ``histogram_file()``, ``correlogram()``,
``correlogram_file()``, ``pacf_correlogram()``, ``pacf_correlogram_file()``.

There is no free-function form: ``.history()`` is the only way to reach an
entity's recorded time series. Report several statistics off the same entity
by chaining ``.history()`` again for each one (``env.q.history().mean()``,
``env.q.history().std()``, ...) -- each call is a cheap native lookup, not a
new recording.

Use ``capture()`` when you need the raw time path in Python without writing a
file. It is declared in the model-level collector:

.. code-block:: python

   class Clinic(sim.Model):
       waiting_room: sim.Queue
       mean_queue_len: sim.Output

       @sim.collect
       def collect_stats(env: "Clinic"):
           env.mean_queue_len = env.waiting_room.history().mean()
           env.waiting_room.history().capture()

After ``exp.run()``, read the captured arrays from the experiment:

.. code-block:: python

   rows = exp.history("waiting_room", trial=0)
   all_rows = exp.histories("waiting_room")

Each returned array is ``float64`` with columns ``time``, ``value``, and
``duration``. Captured names use the flattened field path, such as
``"waiting_room"`` or ``"station__queue"``. The arrays are trial-local and
aligned with the experiment row order.

For entity fields owned by a fixed component collection, the collection
dimension is retained.  For example, capturing
``env.counters[index].line.history()`` makes ``exp.histories("counters__line")``
a tuple of trials, each containing one history array per counter in
collection order.  Select one item directly with
``exp.history("counters__line", trial=0, index=index)``.  The per-item arrays
use the same three columns as scalar captures.

For text reports and text-mode plots, the no-suffix methods above (both
dataset and ``.history()`` methods) print to stdout for console and notebook
use; the ``*_file()`` variants write to a path handle created with
``sim.log_text()`` instead. A dataset report is produced from the current
trial's dataset. In multi-replication experiments, writing every replication
to the same path can interleave or overwrite raw samples depending on the
append flag and parallel execution. These methods are most useful in
single-trial debugging; scalar ``sim.Output`` fields are usually better for
large parallel experiments.

``Model.experiment(..., warmup=..., duration=...)`` controls the measurement
window: warmup lets the model reach a representative state before summaries are
collected.
