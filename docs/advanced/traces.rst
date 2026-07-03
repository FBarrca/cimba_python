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

A callable with two required positional parameters also receives the trial
index -- ``def generator(rng, trial)`` -- for traces that depend on the
trial's position in the design (trial order is design-point-major with
replications innermost). Parameters with defaults do not opt in.

Bootstrap resampling
--------------------

When observed data is available, ``cimba.bootstrap`` provides ready-made
callable generators that resample it instead of assuming a distribution:

* ``iid(data, length)`` -- resamples single observations; only for serially
  independent data, since it destroys autocorrelation.
* ``moving_block(data, length, block)`` and
  ``circular_block(data, length, block)`` -- resample fixed-length contiguous
  blocks, preserving within-block dependence; the circular variant wraps past
  the end of the series so edge observations are not underweighted.
* ``stationary(data, length, mean_block)`` -- blocks with random geometric
  lengths, so the resampled series has no fixed seams. A good default for
  autocorrelated series such as demand histories.
* ``residual(data, length, trend=..., period=...)``, ``wild(data, ...)``, and
  ``sieve(data, length, order=...)`` -- for trending, seasonal, or
  autoregressive data: fit the structure internally, bootstrap what remains,
  and add the structure back. ``nonnegative=True`` clips at zero;
  ``start=len(data)`` simulates the horizon after the history.
* ``intermittent(data, length)`` -- zero-inflated series (spare parts):
  Markov-chain demand occurrence plus resampled nonzero sizes.
* ``joint(panel, length, name=..., mean_block=...)`` -- several correlated
  series resampled with one set of block draws, preserving their
  cross-correlation across separate trace fields.

Each factory validates its arguments up front and returns an ordinary
``f(rng)`` closure, so it plugs directly into a trace field -- or composes
with your own code. Here the clinic resamples observed appointment gaps and
accumulates them into arrival times:

.. code-block:: python

   from cimba import bootstrap

   gap_resampler = bootstrap.stationary(observed_gaps, length=64, mean_block=8)


   def appointment_generator(rng):
       return gap_resampler(rng).cumsum()


   exp = model.experiment(
       appointments=appointment_generator,
       replications=200,
       duration=12.0,
       warmup=0.0,
       seed=42,
   )

Because these are plain callables, everything on this page applies: one
experiment seed reproduces every resample, ``sim.trace_rng`` rebuilds any
trial's trajectory, and ``model.trial_seeds`` supports precomputing rows in
parallel. For the full landscape of methods -- when block bootstraps apply,
handling trend and seasonality, block-length selection, and methods worth
writing by hand -- see :doc:`bootstrapping`.

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
