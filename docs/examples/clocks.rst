Clock Processes
===============

This is the smallest useful Cimba model: a process that periodically wakes up.

.. code-block:: python

   import cimba


   def ticker(me, ticks):
       while True:
           cimba.hold(1.0)
           ticks.append(cimba.time())


   def run(stop_time=3.5, seed=12):
       ticks = []
       with cimba.Simulation(seed=seed) as sim:
           cimba.Process("Ticker", ticker, ticks).start()
           sim.stop_at(stop_time)
           sim.execute()
           return ticks


   print(run())

The process wakes at simulated times 1.0, 2.0, and 3.0. The stop event at 3.5
clears the future wakeup scheduled for 4.0.
