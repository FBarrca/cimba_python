Monitoring and Statistics
=========================

Recording state over time
-------------------------

Buffers, queues, resources, and resource pools can record their state as a
time series:

.. code-block:: python

   queue.start_recording()
   sim.execute()
   queue.stop_recording()

   history = queue.history()
   print(history.values())
   print(history.summary().mean)

The summary of a :class:`cimba.TimeSeries` is duration-weighted, so it is the
right tool for average queue lengths, utilization, and occupancy.

Unweighted samples
------------------

Use :class:`cimba.DataSummary` when every observation has equal weight:

.. code-block:: python

   waits = cimba.DataSummary()
   waits.add(customer_wait)
   print(waits.mean)

Use :class:`cimba.Dataset` when you need to keep all sample values and compute a
median later.

Weighted samples
----------------

Use :class:`cimba.WeightedSummary` when samples have explicit weights:

.. code-block:: python

   summary = cimba.WeightedSummary()
   summary.add(value=5.0, weight=2.0)

Replications
------------

The current Python API focuses on one simulation per Python thread. For
replications, write a function that creates one :class:`cimba.Simulation`, runs
one trial, and returns ordinary Python results.
