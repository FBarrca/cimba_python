Quickstart
==========

This page takes you from ``pip install`` to a working model and a confidence
interval in about five minutes. Each step points to the page that explains it
properly, so you can skim now and dig in later.

1. Install
----------

Cimba Python needs Python 3.13 or newer. Wheels for Linux x86_64, Windows
AMD64 and macOS arm64 contain the native engine, so there is nothing else to
compile.

.. code-block:: bash

   pip install cimba          # or: uv add cimba

Check that the engine loads:

.. code-block:: pycon

   >>> import cimba as cb
   >>> cb.engine_version()
   '3.0.0'

Building from source is covered in :doc:`installation`.

2. Describe the system
----------------------

We will simulate a single-server queue: customers arrive at random, wait in
line, and are served one at a time. Save this as ``queue.py``:

.. code-block:: python
   :linenos:

   import cimba as cb
   from cimba import inputs


   class Queue(cb.Model):
       # Inputs: the variability the model consumes, with default sources.
       interarrival: cb.Input[float] = inputs.dist.exponential(mean=1 / 0.75)
       service_time: cb.Input[float] = inputs.dist.exponential(mean=1.0)

       # An entity: a counted level with time-weighted statistics.
       line: cb.Container

       # An output: one number per trial, written at the end.
       mean_line: cb.Output[float]

       @cb.process
       def arrivals(self):
           while True:
               cb.hold(self.interarrival.next())   # wait for the next customer
               self.line.put(1)                     # they join the line

       @cb.process
       def server(self):
           while True:
               self.line.get(1)                     # blocks while the line is empty
               cb.hold(self.service_time.next())    # serve them

       @cb.on_end
       def measure(self):
           self.mean_line = self.line.mean_level()

Read it top to bottom, the way you would describe the system to a colleague:
*customers arrive every so often and join the line; the server takes one from
the line and serves them for a while; at the end we look at the average line
length.* That's all the model contains.

A few things to notice:

* ``cb.hold()`` and ``self.line.get(1)`` **block**. The process pauses there,
  simulated time moves on, and the process resumes on the next line. You never
  write ``yield`` or a callback.
* ``self.interarrival.next()`` asks the input for its next value. The model
  doesn't know or care whether that value comes from a distribution or from a
  spreadsheet of real arrivals.
* The methods are compiled to native code with Numba. They use a subset of
  Python: numbers, NumPy, loops and the Cimba verbs. See
  :doc:`concepts/compiled_code`.

3. Run an experiment
--------------------

Add this below the class:

.. code-block:: python
   :linenos:
   :lineno-start: 30

   model = Queue()
   experiment = cb.Experiment(
       model,
       replications=50,                                    # 50 independent trials
       window=cb.Window(warmup=100.0, duration=10_000.0),  # measure after a warmup
       seed=2026,                                          # reproducible
   )
   results = experiment.run()                              # uses every core

   summary = cb.analysis.summary(results[model].mean_line)
   print(f"mean line length: {summary.mean[0]:.3f} "
         f"(95% CI {summary.lower[0]:.3f} .. {summary.upper[0]:.3f})")

.. code-block:: console

   $ python queue.py
   mean line length: 2.321 (95% CI 2.256 .. 2.387)

Queueing theory predicts ρ²/(1−ρ) = 2.25 for ρ = 0.75. We're close but not
quite on it, and that's instructive rather than alarming. At high
utilization the queue has long, slow swings, so 10,000 time units per trial
is still fairly short, and the warmup of 100 doesn't fully remove the empty
start. Lengthen the window or add replications and the interval tightens
around 2.25. Tutorial chapter 1 does exactly that across a whole range of
utilizations. The first run takes a second or two longer because the class
is compiled; later runs in the same process reuse the compiled code.

``results[model].mean_line`` is a :class:`~cimba.Samples` object: a read-only
array shaped ``(design points, replications)``, here ``(1, 50)``. You read
results through the **same object** you configured, so there are no string
keys to get wrong.

4. Ask a what-if question
-------------------------

What happens as the server gets busier? Replace the default arrival source with
a **sweep**:

.. code-block:: python

   rho = cb.sweep(0.5, 0.7, 0.9)
   model.interarrival = inputs.dist.exponential(mean=rho.map(lambda r: 1 / r))

   results = cb.Experiment(model, replications=50,
                           window=cb.Window(warmup=100.0, duration=10_000.0),
                           seed=2026).run()

   for level, mean in zip(results.levels(rho),
                          cb.analysis.summary(results[model].mean_line).mean):
       print(f"rho={level:.1f}  mean line={mean:.2f}")

Now there are three design points × 50 replications = 150 trials, run in
parallel. Replication *r* uses the same random numbers at every design point
(*common random numbers*), so differences between points come from the
change you made, not from luck. :func:`cimba.analysis.compare` turns that into
a paired confidence interval.

5. Swap in real data
--------------------

Suppose you logged real inter-arrival gaps. Bind them to the same input:

.. code-block:: python

   import numpy as np

   gaps = np.loadtxt("observed_gaps.csv")

   model.interarrival = inputs.trace(gaps, on_exhausted="wrap")              # replay as recorded
   model.interarrival = inputs.bootstrap.stationary(gaps, mean_block=10)    # or resample it

The class is not recompiled and its code is unchanged. Only the input's
**source** changed. That is the central idea of Cimba Python; the
:doc:`concepts/inputs` page explains it in full.

Where next
----------

* Follow the :doc:`tutorial/index` for a guided tour of processes, resources,
  dynamic entities, conditions and data-driven inputs.
* Read :doc:`concepts/big_picture` for a one-page mental model.
* Keep the :doc:`reference/index` open while you write your own models.
