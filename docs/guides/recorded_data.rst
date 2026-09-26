Drive a model with recorded data
================================

**Goal:** replace a textbook distribution with values you measured, without
touching the model.

1. Make sure the variability is an input
----------------------------------------

The model must consume the value through an ``Input`` or ``Series`` field.
If it currently draws inline, for example
``cb.hold(cb.random.exponential(2.0))``, move the draw into a field first:

.. code-block:: python

   class Clinic(cb.Model):
       arrival_gap: cb.Input[float] = inputs.dist.exponential(mean=2.0)

       @cb.process
       def arrivals(self):
           while True:
               cb.hold(self.arrival_gap.next())
               ...

Use ``Input`` for a sequence consumed event by event (gaps, durations,
sizes). Use ``Series`` for values tied to calendar time (daily demand, hourly
staffing, weather):

.. code-block:: python

   demand: cb.Series[float] = cb.Series(step=1.0)          # day k = bucket k
   staffing: cb.Series[int] = cb.Series(step=60.0, origin=0.0)

2. Bind the recording
---------------------

.. code-block:: python

   import numpy as np
   from cimba import inputs

   gaps = np.loadtxt("arrival_gaps.csv")
   clinic = Clinic()
   clinic.arrival_gap = inputs.trace(gaps)

The array is copied and fingerprinted, so later changes to ``gaps`` can't
leak into the study.

3. Decide what happens at the end of the data
---------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``on_exhausted=``
     - Use when…
   * - ``"fail"`` (default)
     - Running out means the study is invalid. The trial fails with a reason
       naming the input and how many values it consumed.
   * - ``"wrap"``
     - The recording is a representative cycle (a typical week) and
       repeating it is acceptable.
   * - ``"end_trial"``
     - The trial should simply stop when the data ends. In 0.7.0 such
       trials are reported as failed with the exhaustion reason, so treat
       their outputs accordingly.

For a ``Series`` fed by a ``"fail"`` trace, Cimba checks the length against
the window when you create the ``Experiment`` and tells you how many rows are
needed:

.. code-block:: text

   ModelDefinitionError: clinic.demand: trace has 300 rows, needs 365

4. Replay, or resample?
-----------------------

A trace replays one history, and every replication sees the same values.
That's right for **validation** (does the model reproduce last year's
waiting times?), but it gives you one sample of the future, not many. To
estimate variability, **resample** the recording instead. Each replication
gets a new, plausible history with the data's shape:

.. code-block:: python

   clinic.arrival_gap = inputs.bootstrap.iid(gaps)                       # independent values
   clinic.arrival_gap = inputs.bootstrap.stationary(gaps, mean_block=7)  # keeps autocorrelation

Resamples never run out: they are extended transparently (see
:doc:`../concepts/inputs`). Next: :doc:`input_models`.

5. Check what was used
----------------------

.. code-block:: python

   record = results[clinic].arrival_gap
   record.source[0]["method"], record.source[0]["sha256"]
   record.consumed.max()          # the most values any trial needed
   record.rows(0, 0)              # the exact values of trial (0, 0)
