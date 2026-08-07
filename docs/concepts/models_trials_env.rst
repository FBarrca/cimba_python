Models, Trials, and the root ``self`` view
==========================================

A ``sim.Model`` describes the shape of one simulated world. It names the
parameters supplied by an experiment, the outputs collected after each trial,
the mutable state inside a trial, and the passive simulation entities that
processes use to interact.

Conceptually, the model is the root callback/declaration owner around its
component tree. ``sim.Model`` and ``sim.Component`` therefore share callback
discovery, inheritance, validation, and compilation semantics internally.
They remain distinct public classes: a component is a nestable authoring
template addressed with ``self`` inside callbacks, while a model owns the
complete trial namespace and the experiment/result lifecycle. In a Model
callback, ``self`` is the root trial-environment view; it is not the Python
Model instance. Models are not valid nested components.

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

       @sim.process
       def arrivals(self: "Clinic"):
           while True:
               sim.hold(random.exponential(1.0 / self.arrival_rate))
               self.waiting_room.put(1)

   model = Clinic("clinic")

The class is not one patient or one run. It is the declaration for every trial
that the experiment will create. Each trial receives its own ``arrival_rate``,
``mean_service``, outputs, queue, resource, and state counter.

The trial record
----------------

Model process functions receive the current trial as ``self``:

.. code-block:: python

   class Clinic(sim.Model):
       arrival_rate: sim.Param
       waiting_room: sim.Queue

       @sim.process
       def arrivals(self: "Clinic"):
           while True:
               sim.hold(random.exponential(1.0 / self.arrival_rate))
               self.waiting_room.put(1)

The ``self`` view is trial-local. Reading ``self.arrival_rate`` reads the value
for this trial, and ``self.waiting_room`` is the queue handle created for this
trial. Another replication or parameter combination gets a different record and
different native entities.

That separation is the reason Cimba Python can run experiments in parallel.
One trial does not share simulation state with another trial.

Where the class may be declared
-------------------------------

A ``sim.Model`` subclass may be declared anywhere a class may be, including
inside a function -- which is the natural way to write a parameterised
diagnostic or a test helper. Each such class is a distinct declaration, so
building one twice with different sizes gives two independent models; nothing is
keyed on the class's qualified name.

The one restriction comes from Python, not Cimba: a *string* annotation is
resolved against module globals, so it cannot name a class defined inside a
function. That bites whenever the annotation is quoted, and in any module using
``from __future__ import annotations`` it bites for every annotation, because
there every annotation is a string:

.. code-block:: python

   from __future__ import annotations

   def build():
       class Holder(sim.Component):      # local, so invisible to the lookup
           size: sim.Param

       class Line(sim.Model):
           holder: Holder = Holder()     # NameError: cannot resolve 'Holder'

Declare the referenced class at module scope. Cimba reports this as a
``NameError`` naming the annotation it could not resolve, rather than letting
the bare Python error through.

Field roles
-----------

``sim.Param`` fields are inputs. They may be required or carry a scalar
declaration default:

.. code-block:: python

   class Clinic(sim.Model):
       arrival_rate: sim.Param
       mean_service: sim.Param = 0.25

``model.experiment(arrival_rate=4.0)`` uses ``0.25`` for ``mean_service``.
Passing ``mean_service=...`` overrides the default with a scalar or swept
array. Required parameters such as ``arrival_rate`` still produce a missing
parameter error when omitted. ``model.param_defaults`` exposes the effective
flattened defaults.

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

Put model state on ``self`` when a process needs it during the simulated run.
Keep ordinary Python analysis code outside the process body and use it before
or after ``exp.run()``.

This keeps the compiled simulation path focused on simulation behavior, while
the Python side remains free for preparing inputs, summarizing outputs, and
plotting results.
