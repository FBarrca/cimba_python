Reporting and Logging
=====================

Advanced models need two kinds of observability. During model construction,
logging and reports help explain what happened in one trial. During analysis,
``sim.Output`` fields and structured reporting helpers turn many trials into
data.

Debug logging
-------------

Top-level logger flags control native logging. Process-body logging uses stable
text handles created outside the hot loop:

.. code-block:: python

   import cimba
   import cimba.sim as sim

   USER_TRACE = 0x00000001
   MSG_ARRIVED = sim.log_text("patient arrived")
   MSG_WAITING = sim.log_text("waiting room level")


   @model.process
   def arrivals(env: Clinic):
       sim.log_user(USER_TRACE, MSG_ARRIVED)
       sim.log_user_i64(USER_TRACE, MSG_WAITING, env.waiting_room.level())


   cimba.logger_flags_on(USER_TRACE)

Use logging for small runs while validating behavior. Turn it off for large
experiments unless the output is part of the experiment design.

Native text reports
-------------------

Every entity's ``.report()`` method prints a native-style text report for
queues, resources, pools, stores, and priority queues (datasets and time
series have the same ``.report()``/``.report_file()`` pair, described in
:doc:`../api_reference/data`):

.. code-block:: python

   @model.process
   def debug_report(env: Clinic):
       sim.hold(480.0)
       env.waiting_room.report()
       env.doctor.report()

``.report_file(path, append=1)`` writes to a path handle created with
``sim.log_text()``. These reports are most useful for single-trial debugging
and tutorial-style inspection.

Structured Python reports
-------------------------

Outside compiled process code, prefer :mod:`cimba.reporting` when you want
records, tables, or plots:

.. code-block:: python

   from cimba import reporting

   report = reporting.resource_report(trial.queue, lags=20)
   print(reporting.format_report(report))

   rows = report.histogram.to_records()
   summary = report.summary.as_dict()

Structured reports return ordinary Python data that can be handed to pandas,
Polars, CSV writers, notebooks, or plotting code. Plotting helpers are imported
only when used and require the optional plotting extra.

Outputs for experiments
-----------------------

For replicated experiments and parameter sweeps, scalar ``sim.Output`` fields
are usually the best data surface:

.. code-block:: python

   @model.collect
   def collect_stats(env: Clinic):
       env.avg_waiting = env.waiting_room.mean_level()
       env.completed = float(env.served)

Outputs are aligned with the experiment trial table and are easy to group by
parameters. Reports are richer, but they are usually better for diagnosing a
few runs than for summarizing thousands of trials.

Choosing the right surface
--------------------------

Use logging when you need to watch model behavior as it unfolds.

Use native text reports when validating one trial interactively.

Use structured ``cimba.reporting`` helpers when you need report data in Python
after a run.

Use ``sim.Output`` fields when a metric belongs in every trial row of an
experiment.

For reporting helpers see :doc:`../api_reference/data`, and for logging helpers
and constants see :doc:`../api_reference/logging`.
