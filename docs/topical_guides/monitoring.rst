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

For replications, write a trial function that creates one
:class:`cimba.Simulation`, runs one trial, computes per-trial metrics, and
returns ordinary Python results:

.. code-block:: python

   def trial(index, seed):
       with cimba.Simulation(seed=seed) as sim:
           queue = cimba.Buffer("Queue")
           queue.start_recording()
           cimba.Process("Arrival", arrival, queue).start()
           cimba.Process("Service", service, queue).start()
           sim.stop_at(2000.0)
           sim.execute()
           queue.stop_recording()
           return queue.history().summary().mean


   means = cimba.run_experiment(trial, n=100, seed=12345)

   across_replications = cimba.DataSummary()
   for mean in means:
       across_replications.add(mean)

The default ``backend="process"`` is recommended for Python simulations. It
requires return values that can be pickled, so return floats, tuples, or
dictionaries instead of native Cimba statistics objects.

Use ``backend="thread"`` when a trial needs to return native in-process objects
such as :class:`cimba.TimeSeries`, :class:`cimba.DataSummary`, or
:class:`cimba.WeightedSummary`. The thread backend only parallelizes on
free-threaded Python builds; on GIL-enabled interpreters it runs serially.
