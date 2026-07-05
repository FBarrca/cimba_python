Models, Trials, and ``env``
===========================

A ``sim.Model`` describes the shape of one simulated world. It names the
parameters supplied by an experiment, the outputs collected after each trial,
the mutable state inside a trial, and the passive simulation entities that
processes use to interact.

.. code-block:: python

   import cimba.sim as sim

   import cimba.random as random

   class Clinic(sim.Model):
       arrival_rate: sim.Param
       mean_service: sim.Param
       completed: sim.Output
       avg_waiting: sim.Output
       waiting_room: sim.Queue
       doctor: sim.Resource
       served: sim.State

   model = Clinic("clinic")

The class is not one patient or one run. It is the declaration for every trial
that the experiment will create. Each trial receives its own ``arrival_rate``,
``mean_service``, outputs, queue, resource, and state counter.

The trial record
----------------

Process functions receive the current trial as ``env``:

.. code-block:: python

   @model.process
   def arrivals(env: Clinic):
       while True:
           sim.hold(random.exponential(1.0 / env.arrival_rate))
           sim.put(env.waiting_room, 1)

The ``env`` object is trial-local. Reading ``env.arrival_rate`` reads the value
for this trial, and ``env.waiting_room`` is the queue handle created for this
trial. Another replication or parameter combination gets a different record and
different native entities.

That separation is the reason Cimba Python can run experiments in parallel.
One trial does not share simulation state with another trial.

Field roles
-----------

``sim.Param`` fields are inputs. They are set by ``model.experiment(...)`` and
may be scalars or swept arrays.

``sim.Output`` fields are results. They start as missing values and are usually
written by a collector after the trial finishes.

``sim.State`` and ``sim.FloatState`` fields are mutable trial-local variables.
Use them for counters, flags, and numeric state that should reset for every
trial.

Entity fields such as ``sim.Queue``, ``sim.Resource``, ``sim.Pool``,
``sim.Store``, ``sim.Condition``, and ``sim.Dataset`` are handles to native
simulation objects. Declare them on the model so Cimba Python can create,
record, and destroy the right objects for every trial.

A useful rule of thumb
----------------------

Put model state on ``env`` when a process needs it during the simulated run.
Keep ordinary Python analysis code outside the process body and use it before
or after ``exp.run()``.

This keeps the compiled simulation path focused on simulation behavior, while
the Python side remains free for preparing inputs, summarizing outputs, and
plotting results.
