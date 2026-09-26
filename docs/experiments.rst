Experiments and results
=======================

Configure model objects directly. Use ``cimba.sweep(*values)`` for an
independent design axis; use ``cimba.sweeps(*columns)`` for linked axes.
Parameters and input sources may be swept. Then create an experiment:

.. code-block:: python

   import cimba as cb

   model.base_stock = cb.sweep(250.0, 300.0, 350.0)
   result = cb.Experiment(
       model, replications=100,
       window=cb.Window(warmup=30.0, duration=365.0),
       seed=7,
   ).run(workers=4)
   samples = result[model].service_level
   report = cb.analysis.summary(samples)
   comparison = cb.analysis.compare(samples, a=0, b=2)

``Samples.values`` is shaped ``(design points, replications)``. Failed
trials appear in ``result.failed`` with reasons in
``result.failure_reasons``. ``result[model].input_name`` contains source
provenance and per-trial consumption. Call ``entity.capture()`` before a run
to retain its history. Results and their arrays are read-only.
