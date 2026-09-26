Migrating from 0.6
==================

Version 0.7 replaces the Python modeling core; the Cimba engine is
unchanged. The old ``cimba.sim`` and ``cimba.bootstrap`` imports raise an
``ImportError`` that points here, so a half-migrated model can't run by
accident. The full guide is
`MIGRATION.md on GitHub <https://github.com/FBarrca/cimba_python/blob/master/MIGRATION.md>`_.
This table covers the most common changes:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.6
     - 0.7
   * - ``import cimba.sim as sim``
     - ``import cimba as cb``
   * - ``sim.Model``, ``Component``, ``Struct``, spawnable processes
     - ``cb.Model`` for everything; ``cb.spawn(Class, ...)`` during a trial
   * - ``sim.Trace`` field and a manual cursor
     - ``cb.Input[T]`` with ``next()``; ``cb.Series[T]`` with ``now()``/``at(t)``
   * - ``random.*`` for arrivals, services, demand
     - an input bound to ``inputs.dist.*``
   * - ``cimba.bootstrap.*(data, length=...)``
     - ``inputs.bootstrap.*(data, ...)``; lengths are managed for you
   * - ``trace_rng_name``
     - ``inputs.bootstrap.joint(panel, ...)[key]`` or a shared ``tag``
   * - trace arrays passed to ``experiment(field=...)``
     - assign the source on the model object
   * - exhaustion tripwire outputs
     - ``on_exhausted=`` policies; resamples extend automatically
   * - ``sim.Param``/``Output``/``State``/``FloatState``/``Const``
     - ``cb.Param[T]``, ``cb.Output[T]``, ``cb.State[T]``, plain constants
   * - ``Queue``, opaque ``Store``, ``f2i``/``i2f``, ``Pool``
     - ``Container``, typed ``Store[T]``/``PriorityStore[T]``,
       ``Resource(capacity=n)``
   * - ``Predicate``/``Event``/``Processes`` fields with ``field=``
     - decorated methods referenced directly: ``wait_until(self.ready)``
   * - ``@collect``, an initialization process, ``@function``
     - ``@cb.on_end``, ``@cb.on_start``, a module-level ``numba.njit``
       helper taking the model view
   * - ``model.experiment(**params)``, flattened names, ``{i: v}`` dicts
     - configure objects, then ``cb.Experiment(model, replications=,
       window=, seed=)``
   * - ``exp.trials``, ``exp["x"]``, result namespaces
     - ``results[model].x`` (immutable, object-indexed)
   * - in-trial reports
     - ``cimba.analysis`` after the run; ``cb.log`` for tracing

Every tutorial script uses the 0.7 API. :doc:`tutorial/index` is a good way to
relearn the library quickly.
