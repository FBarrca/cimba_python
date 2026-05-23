M/M/1 Queue
===========

This model has one arrival process, one service process, and one
:class:`cimba.Buffer` representing the queue.

.. code-block:: python

   from dataclasses import dataclass

   import cimba


   @dataclass
   class Trial:
       arrival_rate: float = 0.75
       service_rate: float = 1.0
       duration: float = 1000.0
       arrivals: int = 0
       services: int = 0
       average_queue_length: float = 0.0


   def arrival(me, trial):
       mean = 1.0 / trial.arrival_rate
       while True:
           cimba.hold(cimba.exponential(mean))
           trial.arrivals += 1
           trial.queue.put(1)


   def service(me, trial):
       mean = 1.0 / trial.service_rate
       while True:
           trial.queue.get(1)
           cimba.hold(cimba.exponential(mean))
           trial.services += 1


   def recorder(me, trial):
       trial.queue.start_recording()
       cimba.hold(trial.duration)
       trial.queue.stop_recording()
       trial.arrival_process.stop()
       trial.service_process.stop()
       trial.simulation.clear()


   def run(seed=123):
       trial = Trial()
       with cimba.Simulation(seed=seed) as sim:
           trial.simulation = sim
           trial.queue = cimba.Buffer("Queue")
           trial.arrival_process = cimba.Process("Arrival", arrival, trial).start()
           trial.service_process = cimba.Process("Service", service, trial).start()
           cimba.Process("Recorder", recorder, trial).start()
           sim.execute()
           trial.average_queue_length = trial.queue.history().summary().mean
       return trial


   print(run().average_queue_length)

The examples in ``tutorial/tut_1_1.py`` through ``tutorial/tut_1_7.py`` build
this model up in smaller steps.
