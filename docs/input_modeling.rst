Input modeling
==============

Declare a sequence as ``Input[T]`` and consume it with ``next()``. Declare
time-indexed data as ``Series[T] = Series(step=..., origin=...)`` and read it
with ``now()`` or ``at(t)``. Assign a source on the model object before
creating an experiment:

.. code-block:: python

   from cimba import inputs

   model.interarrival = inputs.dist.exponential(mean=1.5)
   model.interarrival = inputs.trace(recorded_gaps, on_exhausted="wrap")
   model.interarrival = inputs.bootstrap.stationary(recorded_gaps, mean_block=7)

Recorded traces support ``fail``, ``wrap``, and ``end_trial`` exhaustion
policies. Resampled and fitted sources extend by regenerating a longer,
prefix-stable row. Each input owns its own random stream, so consumption of
another input does not change its values.

For observed data, ``inputs.fit(data)`` ranks supported distribution fits.
``cimba.analysis.check_input(source, reference=data)`` compares means,
variance, quantiles, autocorrelation, partial autocorrelation, and a KS
statistic before a run.

The ``tutorial/multi_echelon_inventory.py`` example uses a joint stationary
bootstrap to preserve dependence among five daily demand series. The result
exposes source descriptions, consumed counts, extension counts, and replayed
rows through ``results[facility].demand``.
Pass the joint source and a matching reference panel to
``analysis.check_input`` to compare their cross-correlation matrices.
