Shared Entities
===============

Processes become useful when they interact through shared simulation entities.
Declare those entities on the model so every trial gets its own queue,
resource, pool, condition, or dataset.

Choosing an entity
------------------

Use ``sim.Queue`` for numeric amounts: patients waiting, jobs in a backlog,
inventory units, tokens, or fluid-like quantities.

Use ``sim.Resource`` for one exclusive server. A process acquires it, holds it
while service happens, and releases it.

Use ``sim.Pool`` for multiple interchangeable units of capacity, such as a
group of three nurses or ten identical machines.

Use ``sim.Store`` when queued items have identity encoded as integer payloads.
For example, a process can place an order id in a store and another process can
take that same id later.

Use ``sim.Condition`` when readiness depends on model-specific state that does
not fit a queue or resource operation.

Use ``sim.Dataset`` for untimed samples collected during the run, such as
service durations or observed wait times.

Queues and resources
--------------------

A queue models waiting work:

.. code-block:: python

   import cimba.random as random

   class Clinic(sim.Model):
       arrival_rate: sim.Param
       mean_service: sim.Param
       waiting_room: sim.Queue

       @sim.process
       def arrivals(self: "Clinic"):
           while True:
               sim.hold(random.exponential(1.0 / self.arrival_rate))
               self.waiting_room.put(1)

       @sim.process
       def service(self: "Clinic"):
           while True:
               self.waiting_room.get(1)
               sim.hold(random.exponential(self.mean_service))

A resource models exclusive access:

.. code-block:: python

   class Clinic(sim.Model):
       mean_service: sim.Param
       doctor: sim.Resource

       @sim.process
       def patient(self: "Clinic"):
           self.doctor.acquire()
           try:
               sim.hold(random.exponential(self.mean_service))
           finally:
               self.doctor.release()

The queue version is natural when patients are just a count. The resource
version is natural when each patient process carries its own path through the
model and must wait for a doctor.

Pools and variable capacity
---------------------------

Use a pool when there are several interchangeable units:

.. code-block:: python

   class Clinic(sim.Model):
       doctor_count: sim.Param
       doctors: sim.Pool = sim.capacity("doctor_count")

       completed: sim.Output
       served: sim.State

An experiment can sweep ``doctor_count`` just like other parameters. Use
integer-valued parameter values when a parameter controls capacity.

Conditions and datasets
-----------------------

A condition is useful when a process waits on a predicate over model state:

.. code-block:: python

   class Clinic(sim.Model):
       open: sim.State
       shift_started: sim.Condition
       ready: sim.Predicate

       @sim.predicate(field="ready")
       def is_open(self: "Clinic") -> bool:
           return self.open == 1

       @sim.process
       def late_staff(self: "Clinic"):
           self.shift_started.wait_for(self.ready)

       @sim.process
       def manager(self: "Clinic"):
           self.open = 1
           self.shift_started.signal()

A dataset collects samples:

.. code-block:: python

   service_time = random.exponential(env.mean_service)
   env.service_times.add(service_time)
   sim.hold(service_time)

Use entity summaries, such as ``env.waiting_room.mean_level()``, for
time-weighted measurements. Use datasets for samples that happen at individual
moments.

For deeper API details, see :doc:`../api_reference/entities`.
