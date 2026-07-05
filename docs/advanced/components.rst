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

Primitive per-instance settings can be marked explicitly with ``sim.Const``.
Constants are captured from each component instance at model construction time
and lowered as compile-time values or small lookup tables:

.. code-block:: python

   class Desk(sim.Component):
       server_count: sim.Const[int]
       waiting: sim.Queue

       def __init__(self, server_count: int):
           self.server_count = server_count

       @sim.process(copies="server_count")
       def clerk(self, env, idx):
           ...

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

Wiring components together
--------------------------

Routing between components can be declared where the components are declared.
Accessing a declared ``Queue``/``Resource``/``Pool``/``Store``/``Condition``
field on a component instance yields a wiring reference; passing it as another
instance's same-kind field value makes both fields name the same entity:

.. code-block:: python

   class Station(sim.Component):
       inbox: sim.Store
       outbox: sim.Store

       def __init__(self, mean_time: float, *, inbox=None):
           self.mean_time = mean_time
           if inbox is not None:
               self.inbox = inbox

       @sim.process
       def server(self, env):
           while True:
               item = sim.store_take(self.inbox)
               sim.hold(sim.exponential(self.mean_time))
               sim.store_put(self.outbox, item)


   class AssemblyLine(sim.Model):
       station_1: Station = Station(5.0)
       station_2: Station = Station(7.0, inbox=station_1.outbox)
       station_3: Station = Station(4.0, inbox=station_2.outbox)

Here ``station_2.inbox`` is an alias for ``station_1.outbox``: only one store
is created, ``station_1``'s server feeds it, and ``station_2``'s server takes
from it, so parts flow down the line without hand-written routing processes.
Unwired fields (the first inbox, the last outbox) stay ordinary stores that
model-level processes can feed and drain.

Wiring targets must be declared somewhere on the model, both fields must have
the same kind, and wired fields do not appear in the trial table (use the
target's flattened name, e.g. ``station_1__outbox``). Wiring chains are resolved
to the final target after the component tree is built. Component collections
cannot be wired yet.

Routing with component references
---------------------------------

Wiring merges two fields into one entity, which fixes the flow at declaration
time. When the *code* must choose a target — sequences, by-condition
transfer to one of several stations — declare a component reference with
``sim.Ref`` or an indexable reference table with ``sim.Refs``:

.. code-block:: python

   class Station(sim.Component):
       inbox: sim.Store
       downstream: sim.Ref["Station"]

       def __init__(self, mean_time: float, downstream=None):
           self.mean_time = mean_time
           if downstream is not None:
               self.downstream = downstream

       @sim.process
       def server(self, env):
           while True:
               item = sim.store_take(self.inbox)
               sim.hold(sim.exponential(self.mean_time))
               sim.store_put(self.downstream.inbox, item)

The reference value is another component instance declared on the model.
Inside compiled code, ``self.downstream.inbox`` resolves to the target's
fields, constants, and processes exactly as if they were accessed through
their own path. Unlike wiring, references are resolved after the whole model
class is processed, so the target may be declared *after* the component that
references it (values can also be attached post-declaration, e.g.
``Line.station_1.downstream = Line.station_2`` before instantiating).

``sim.Refs`` declares a routing table for runtime decisions. All entries must
be items of a single component collection, so the lookup lowers to array
indexing:

.. code-block:: python

   class Dispatcher(sim.Component):
       inbox: sim.Store
       routes: sim.Refs[Station]

       def __init__(self, routes=()):
           self.routes = tuple(routes)

       @sim.process
       def route(self, env):
           while True:
               item = sim.store_take(self.inbox)
               sim.store_put(self.routes[item % 3].inbox, item)


   class Shop(sim.Model):
       stations: list[Station] = [Station(5.0), Station(7.0), Station(4.0)]
       dispatch: Dispatcher = Dispatcher(
           routes=(stations[0], stations[1], stations[2]))

Model callbacks can follow references too (``env.dispatch.routes[1].inbox``,
``env.stations[j].downstream.inbox``). A fixed ``sim.Ref`` may target any
declared component; following a reference under a *dynamic* collection index
requires every item to reference the same component declaration. Component
methods that need mixed per-instance targets are lowered per instance instead.

Prefer wiring for fixed linear flows (one shared entity, no extra hop) and
references when the model routes items among alternatives at runtime.

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

For the complete API surface, see :doc:`../api_reference/models`.
