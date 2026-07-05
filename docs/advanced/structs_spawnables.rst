Structs and Spawnables
======================

Core models often treat arrivals as counts. Agent-heavy models need each
arrival to carry its own state: arrival time, priority, service choice,
patience, or accumulated outcomes. Use ``sim.Struct`` for per-process fields
and ``sim.Spawnable`` for processes created during a trial.

Per-process fields
------------------

A ``sim.Struct`` declares fields stored with a process. Fields may be ``int``
or ``float``:

.. code-block:: python

   class Patient(sim.Struct):
       arrival: float
       acuity: int
       wait_started: float


   class Clinic(sim.Model):
       patients: sim.Spawnable
       completed: sim.State


   model = Clinic("clinic")


   @model.process
   def patients(env: Clinic, patient: Patient):
       patient.wait_started = sim.now()
       # The process can use its own fields throughout the visit.

The final annotated process parameter receives the current process's struct
view. Multi-copy static processes can also receive a copy index before the
struct view.

Dynamic process creation
------------------------

Declare a ``sim.Spawnable`` field with the same name as the process that should
be created dynamically:

.. code-block:: python

   import cimba.random as random

   @model.process
   def arrivals(env: Clinic):
       while True:
           sim.hold(random.exponential(1.0 / env.arrival_rate))
           handle = sim.spawn(env.patients, env)
           patient = Patient(handle)
           patient.arrival = sim.now()
           patient.acuity = 1 if random.uniform() < 0.2 else 0

The spawned process begins only after the current process blocks. That gives
the spawning process a clean initialization window: create the process, write
its struct fields through the handle, and then let simulated time continue.

Joining and reclaiming
----------------------

``sim.wait_process(handle)`` waits for a spawned process to finish.
``sim.despawn(handle)`` reclaims a finished spawned process:

.. code-block:: python

   @model.process
   def arrivals(env: Clinic):
       handle = sim.spawn(env.patients, env)
       Patient(handle).arrival = sim.now()
       sim.wait_process(handle)
       sim.despawn(handle)

Long-running models should reclaim finished dynamic processes when they are no
longer needed. A common pattern is to put finished handles into a ``sim.Store``
and have a cleanup process despawn them:

.. code-block:: python

   class Clinic(sim.Model):
       patients: sim.Spawnable
       departures: sim.Store


   @model.process
   def cleanup(env: Clinic):
       while True:
           handle = sim.store_take(env.departures)
           sim.despawn(handle)


   @model.process
   def patients(env: Clinic, patient: Patient):
       # ... patient journey ...
       sim.store_put(env.departures, sim.current())

Leftover spawned processes are stopped and reclaimed at the end of a trial, but
explicit cleanup keeps long trials from accumulating completed agents.

Component-owned spawnables
--------------------------

Components can own spawnable fields. This keeps dynamic agents close to the
subsystem that creates them:

.. code-block:: python

   class Intake(sim.Component):
       patient: sim.Spawnable

       @sim.process
       def arrivals(self, env):
           handle = sim.spawn(self.patient, env)
           Patient(handle).arrival = sim.now()

       @sim.process
       def patient(self, env, p: Patient):
           sim.hold(random.exponential(env.mean_service))

Use this when the dynamic process is naturally part of a component. Use a
model-level ``sim.Spawnable`` when the dynamic process crosses many domains.

For a larger worked example with dynamic agents and resources, see
:doc:`../tutorial`.
