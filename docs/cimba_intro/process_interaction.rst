Process Interaction
===================

Processes interact by waiting, resuming, interrupting, and using shared
simulation objects.

Waiting for another process
---------------------------

One process can wait for another to finish:

.. code-block:: python

   def child(me, ctx):
       cimba.hold(2.0)
       return "done"

   def parent(me, ctx):
       proc = cimba.Process("Child", child).start()
       signal = proc.wait()
       ctx["signal"] = signal
       ctx["value"] = proc.exit_value()

   with cimba.Simulation(seed=123) as sim:
       result = {}
       cimba.Process("Parent", parent, result).start()
       sim.execute()

Interrupts and timers
---------------------

Processes can be resumed by other processes, interrupted with a non-success
signal, or given timers:

.. code-block:: python

   def sleeper(me, ctx):
       signal = cimba.yield_process()
       ctx.append(signal)

   with cimba.Simulation(seed=123) as sim:
       signals = []
       proc = cimba.Process("Sleeper", sleeper, signals).start()
       proc.timer_add(5.0)
       sim.execute()

Signals such as ``SUCCESS``, ``INTERRUPTED``, ``STOPPED``, and ``TIMEOUT`` are
exported from :mod:`cimba`.

Ending a run
------------

For simple runs, use :meth:`cimba.Simulation.stop_at`. For model-specific
shutdown logic, use a normal process:

.. code-block:: python

   def stop_model(me, ctx):
       cimba.hold(ctx.duration)
       ctx.arrivals.stop()
       ctx.service.stop()
       ctx.simulation.clear()

``clear()`` removes future events, so the current ``execute()`` call finishes
once control returns to the dispatcher.
