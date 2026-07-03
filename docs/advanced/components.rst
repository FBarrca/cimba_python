Components
==========

``sim.Component`` groups related declarations and process methods. It is useful
when a model has repeated subsystems, such as several clinic desks, triage
areas, departments, or teams that each own their own queues and workers.

.. code-block:: python

   import cimba.sim as sim

   class Desk(sim.Component):
       waiting: sim.Queue
       completed: sim.State

       @sim.process
       def clerk(self, env):
           while True:
               sim.get(self.waiting, 1)
               sim.hold(sim.exponential(env.mean_service))
               self.completed += 1


   class Clinic(sim.Model):
       mean_service: sim.Param
       served: sim.Output
       front_desk: Desk = Desk()

Component process methods use top-level ``@sim.process``. When a model is
constructed, Cimba Python lowers those methods into ordinary model processes.
Inside the method, ``self.waiting`` and ``self.completed`` refer to the
component-owned fields for this trial.

Component-owned statistics
--------------------------

Components can also own their statistics collection. A method marked with
top-level ``@sim.collect`` takes ``(self, env)`` and runs once per instance
at the end of each trial, typically assigning the component's declared
``sim.Output`` fields:

.. code-block:: python

   class Desk(sim.Component):
       waiting: sim.Queue
       avg_queue: sim.Output

       @sim.process
       def clerk(self, env):
           while True:
               sim.get(self.waiting, 1)
               sim.hold(sim.exponential(env.mean_service))

       @sim.collect
       def desk_stats(self, env):
           self.avg_queue = sim.mean_level(self.waiting)

Every instance of a component collection runs its own collect, so per-desk
outputs land in per-instance output slots. Component collects run before the
model-level ``@model.collect`` callback, which can therefore aggregate over
the component outputs:

.. code-block:: python

   @model.collect
   def clinic_stats(env: Clinic):
       env.worst_queue = env.desks[0].avg_queue
       for i in range(1, 3):
           if env.desks[i].avg_queue > env.worst_queue:
               env.worst_queue = env.desks[i].avg_queue

Nested components
-----------------

Components can own other components. This lets the model declaration become a
table of contents for the simulated world:

.. code-block:: python

   class StaffTeam(sim.Component):
       capacity: sim.Pool = sim.capacity("staff_count")


   class Intake(sim.Component):
       line: sim.Queue
       staff: StaffTeam = StaffTeam()


   class Clinic(sim.Model):
       staff_count: sim.Param
       intake: Intake = Intake()

Model code can read nested paths such as ``env.intake.staff.capacity``. The
trial table stores flattened fields internally, but process source can stay
close to the domain structure.

Component collections
---------------------

Use a ``list[ComponentType]`` declaration for fixed repeated subsystems:

.. code-block:: python

   class Clinic(sim.Model):
       mean_service: sim.Param
       desks: list[Desk] = [Desk(), Desk(), Desk()]


   @model.process
   def router(env: Clinic):
       target = 0
       best = sim.level(env.desks[0].waiting)
       for i in range(1, 3):
           length = sim.level(env.desks[i].waiting)
           if length < best:
               target = i
               best = length
       sim.put(env.desks[target].waiting, 1)

The collection length is fixed by the model class. This is a good fit for
known departments, stations, gates, or desks. If the number of active entities
changes during a trial, use dynamic processes instead.

Flattened outputs and trial data
--------------------------------

Component fields are flattened in experiment arrays with ``__`` separators.
For example, ``env.front_desk.completed`` becomes a trial field named
``front_desk__completed``. Most process code should use the natural component
path, while analysis code may sometimes inspect the flattened field names in
``exp.trials``.

Use components to express model structure, not to hide model behavior. If a
component method needs many details from unrelated components, move that
coordination to a model-level process or split the model into clearer domains.

For the complete API surface, see :doc:`../api/cimba`.
