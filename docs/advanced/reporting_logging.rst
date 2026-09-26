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


   class Clinic(sim.Model):
       waiting_room: sim.Queue

       @sim.process
       def arrivals(self: "Clinic"):
           sim.log_user(USER_TRACE, MSG_ARRIVED)
           sim.log_user_i64(USER_TRACE, MSG_WAITING, self.waiting_room.level())


   cimba.logger_flags_on(USER_TRACE)

Use logging for small runs while validating behavior. Turn it off for large
experiments unless the output is part of the experiment design.

Keep text handles as scalar constants or tuples. Global Python lists and
dictionaries follow Numba's usual restrictions; Cimba does not reconstruct
them inside callbacks. Select a handle from such a container in Python before
using it in compiled code.

Native text reports
-------------------

Every entity's ``.report()`` method prints a native-style text report for
queues, resources, pools, stores, and priority queues. Datasets and time
series instead expose ``.print()``, ``.fivenum()``, ``.histogram()``, and their
file variants, described in :doc:`../api_reference/data`:

.. code-block:: python

   class Clinic(sim.Model):
       waiting_room: sim.Queue
       doctor: sim.Resource

       @sim.process
       def debug_report(self: "Clinic"):
           sim.hold(480.0)
           self.waiting_room.report()
           self.doctor.report()

``.report_file(path, append=1)`` writes to a path handle created with
``sim.log_text()``. These reports are most useful for single-trial debugging
and tutorial-style inspection.

Retaining data for Python analysis
----------------------------------

Call ``self.waits.capture()`` or ``self.waiting_room.history().capture()`` in
the model's ``@sim.collect`` callback to retain raw samples or time histories.
After ``exp.run()``, read their NumPy arrays with ``exp.dataset("waits")`` or
``exp.history("waiting_room")``. These arrays can be passed to Python analysis,
export, or plotting tools. Native entity handles expire when a trial ends.

Outputs for experiments
-----------------------

For replicated experiments and parameter sweeps, scalar ``sim.Output`` fields
are usually the best data surface:

.. code-block:: python

   class Clinic(sim.Model):
       avg_waiting: sim.Output
       completed: sim.Output
       waiting_room: sim.Queue
       served: sim.State

       @sim.collect
       def collect_stats(self: "Clinic"):
           self.avg_waiting = self.waiting_room.mean_level()
           self.completed = float(self.served)

Outputs are aligned with the experiment trial table and are easy to group by
parameters. Reports are richer, but they are usually better for diagnosing a
few runs than for summarizing thousands of trials.

Choosing the right surface
--------------------------

Use logging when you need to watch model behavior as it unfolds.

Use native text reports when validating one trial interactively.

Use captures when you need raw sample arrays in Python after a run.

Use ``sim.Output`` fields when a metric belongs in every trial row of an
experiment.

For reporting helpers see :doc:`../api_reference/data`, and for logging helpers
and constants see :doc:`../api_reference/logging`.
