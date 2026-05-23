Single Server Resource
======================

Use :class:`cimba.Resource` when a process must acquire exclusive access before
continuing.

.. code-block:: python

   import cimba


   def job(me, model):
       arrived = cimba.time()
       model["server"].acquire()
       try:
           model["waits"].add(cimba.time() - arrived)
           cimba.hold(model["service_time"])
       finally:
           model["server"].release()


   def source(me, model):
       for i in range(model["jobs"]):
           cimba.Process(f"Job {i}", job, model).start()
           cimba.hold(model["interarrival_time"])


   with cimba.Simulation(seed=123) as sim:
       model = {
           "server": cimba.Resource("Server"),
           "waits": cimba.DataSummary(),
           "jobs": 5,
           "interarrival_time": 1.0,
           "service_time": 2.0,
       }
       cimba.Process("Source", source, model).start()
       sim.execute()
       print(model["waits"].mean)

The server resource wakes the next waiting process when the current holder
releases it.
