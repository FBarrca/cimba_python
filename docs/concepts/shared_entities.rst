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

   @model.process
   def arrivals(env: Clinic):
       while True:
           sim.hold(sim.exponential(1.0 / env.arrival_rate))
           sim.put(env.waiting_room, 1)

   @model.process
   def service(env: Clinic):
       while True:
           sim.get(env.waiting_room, 1)
           sim.hold(sim.exponential(env.mean_service))

A resource models exclusive access:

.. code-block:: python

   @model.process
   def patient(env: Clinic):
       sim.acquire(env.doctor)
       try:
           sim.hold(sim.exponential(env.mean_service))
       finally:
           sim.release(env.doctor)

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
       is_open: sim.Predicate

   @model.predicate
   def is_open(env: Clinic) -> bool:
       return env.open == 1

   @model.process
   def late_staff(env: Clinic):
       sim.wait_for(env.shift_started, env.is_open, env)

   @model.process
   def manager(env: Clinic):
       env.open = 1
       sim.signal(env.shift_started)

A dataset collects samples:

.. code-block:: python

   service_time = sim.exponential(env.mean_service)
   sim.tally(env.service_times, service_time)
   sim.hold(service_time)

Use entity summaries, such as ``sim.mean_level(env.waiting_room)``, for
time-weighted measurements. Use datasets for samples that happen at individual
moments.

For deeper API details, see :doc:`../topical_guides/queues_resources` and
:doc:`../api/cimba`.
