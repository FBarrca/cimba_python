Run big experiments fast
========================

Cimba Python runs model code natively and trials in parallel, so most models
are fast without tuning. When experiments get large, these are the levers.

Use all cores (the default)
---------------------------

``run()`` uses every core. Limit it with ``run(workers=n)`` on a shared
machine. Results don't depend on ``workers``, so pick whatever is convenient.

Compile once
------------

The first ``run()`` of a class compiles it, typically about a second per
class. After that, the class is cached for the life of the Python process,
whatever you change about the configuration. In a notebook, keep the kernel
alive between experiments. In scripts, run several experiments in one process
rather than one process per experiment. ``results.meta.compile`` shows the
compile time of each run.

Keep the hot loop lean
----------------------

* Prefer in-trial summaries (``mean_level()``, ``sample_mean()``) to
  captures. Capture only what you're investigating.
* Leave ``cb.log`` calls in place but disabled; they are cheap when off.
* Bind recorded or resampled data as inputs rather than reading large
  module-level arrays by index. Rows are replayed from compact per-replication
  buffers.

Size replications and windows deliberately
------------------------------------------

The width of a confidence interval shrinks with the square root of the
number of replications. Use a pilot run to estimate the standard deviation,
then pick the replication count for the precision you need. For steady-state
questions, a long window with fewer replications is often cheaper than many
short ones, but you need a warmup long enough to forget the empty start.

Common random numbers are free precision: compare designs *within* one
experiment, or across experiments with the same seed, and use
:func:`~cimba.analysis.compare`.

Memory for data-driven inputs
-----------------------------

Rows for resampled and fitted inputs are generated per replication and shared
across design points. If all replications' rows won't fit in
``input_memory`` (1 GiB by default), replications are processed in chunks
automatically, with identical results. Lower the limit to reduce peak memory:

.. code-block:: python

   experiment.run(input_memory=256 * 2**20)   # 256 MiB of input rows at a time

``results.meta.chunks`` reports how many chunks were used.

Many structural variants
------------------------

Because compiled code depends only on classes, building 50 differently wired
networks from the same classes compiles nothing new after the first. Loop
over variants freely.
