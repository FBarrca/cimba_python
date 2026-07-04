Clock Process
=============

This is the smallest useful Cimba Python model: a process that wakes up on a
fixed period. It shows the essential shape of every model — a
:class:`~cimba.sim.Model` subclass with typed fields, a process function, a
collect function, and an experiment.

.. code-block:: python

   import cimba.sim as sim


   class Clock(sim.Model):
       tick_count: sim.Output   # scalar result reported per trial
       ticks: sim.State         # trial-local counter, auto-zeroed


   model = Clock("Clock")


   @model.process
   def ticker(env: Clock):
       while True:
           sim.hold(1.0)
           env.ticks = env.ticks + 1


   @model.collect
   def collect(env: Clock):
       env.tick_count = env.ticks


   def main() -> None:
       exp = model.experiment(replications=1, duration=3.5, warmup=0.0, seed=12)
       exp.run()
       print(int(exp["tick_count"][0]))


   if __name__ == "__main__":
       main()

The ticker wakes at simulated times 1.0, 2.0, and 3.0. The trial ends at 3.5,
before the wakeup that would have happened at 4.0, so ``tick_count`` is ``3``.
