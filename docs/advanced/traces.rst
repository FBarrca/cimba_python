Data-Driven Traces
==================

``sim.Trace`` lets a trial replay data prepared outside the simulation. Use it
for appointment schedules, demand trajectories, recorded arrivals, bootstrap
resamples, or fitted-generator output.

Declaring and reading a trace
-----------------------------

Declare a trace field on the model and pass trace data to
``model.experiment(...)``:

.. code-block:: python

   class Clinic(sim.Model):
       appointments: sim.Trace
       completed: sim.Output


   model = Clinic("clinic")


   @model.process
   def appointment_arrivals(env: Clinic):
       times = sim.Trace(env.appointments)
       previous = 0.0
       for at in times:
           sim.hold(at - previous)
           previous = at
           # Create or queue the appointment.

Inside compiled model code, ``sim.Trace(env.appointments)`` returns a
read-only float64 view that supports length, indexing, slicing, and iteration.

Trace input shapes
------------------

A 1-D array is shared by every trial:

.. code-block:: python

   exp = model.experiment(
       appointments=[8.0, 8.5, 9.0, 10.0],
       replications=10,
       duration=12.0,
       warmup=0.0,
   )

A 2-D array maps row ``i`` to trial ``i``. Trial order is design-point-major
with replications innermost.

A sequence of 1-D arrays gives ragged per-trial traces, useful when each day
has a different number of appointments.

Callable generators
-------------------

A trace value can be a callable that returns one 1-D array per trial:

.. code-block:: python

   def appointment_generator(rng):
       gaps = rng.exponential(0.25, size=32)
       return gaps.cumsum()


   exp = model.experiment(
       appointments=appointment_generator,
       replications=100,
       duration=12.0,
       warmup=0.0,
       seed=42,
   )

The callable receives a NumPy generator derived from the trial seed and the
trace field name. Passing the same experiment seed reproduces both simulation
streams and generated traces.

Rebuilding or precomputing traces
---------------------------------

Use ``sim.trace_rng(trial_seed, field_name)`` to rebuild the trace generator
for a recorded trial:

.. code-block:: python

   row = appointment_generator(
       sim.trace_rng(exp["seed"][trial_index], "appointments")
   )

For expensive generators, precompute trace rows outside Cimba Python and pass
the finished rows to ``experiment()``:

.. code-block:: python

   seeds = model.trial_seeds(seed=42, replications=100)
   rows = [
       appointment_generator(sim.trace_rng(seed, "appointments"))
       for seed in seeds
   ]
   exp = model.experiment(
       appointments=rows,
       replications=100,
       duration=12.0,
       warmup=0.0,
       seed=42,
   )

Precomputed rows should use the same seeds and field names as the experiment
to stay bit-identical to callable generation.

Trace exhaustion
----------------

When a process consumes its trace and returns, the trial still runs to the
configured warmup, duration, and cooldown. If exhaustion time matters, record
it in an output or state field:

.. code-block:: python

   env.exhausted_at = sim.now()

Generate traces that cover the simulated window, or make exhaustion an explicit
part of the model logic.

For exact validation rules and accepted shapes, see :doc:`../api/cimba`.
