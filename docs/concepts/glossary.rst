Glossary
========

.. glossary::
   :sorted:

   Model
      A subclass of :class:`cimba.Model` describing one part of the system:
      its fields and its behavior. The model passed to an experiment is the
      *root*; models contain other models.

   Static instance
      A model instance that is part of the tree built before the experiment.
      It exists for the whole trial.

   Dynamic instance
      A model instance created during a trial with ``cb.spawn`` and removed
      with ``cb.release``. It can hold values, references, inputs and
      behavior, but no entities or children.

   Field
      A typed, per-instance value declared by an annotation on a model class.

   Param
      ``cb.Param[T]``: a value fixed for a trial. It can be swept.

   State
      ``cb.State[T]``: a value that changes during a trial and starts from
      its initial value in every trial.

   Output
      ``cb.Output[T]``: a per-trial result, collected as :term:`Samples`.

   Constant
      A plain annotated scalar field (``lane: int = 0``), fixed per instance
      and never swept.

   Ref
      ``cb.Ref[M]``: a reference to another model instance in the tree.

   Collection
      A ``list[M]`` (owned children) or ``list[cb.Ref[M]]`` (references)
      field.

   Entity
      A native synchronization object: ``Container``, ``Store``,
      ``PriorityStore``, ``Resource``, ``Condition`` or ``Dataset``.

   Process
      A method decorated with ``@cb.process``, running as a stackful
      coroutine in simulated time. It can block anywhere.

   Hook
      A method decorated with ``@cb.on_start`` (before processes) or
      ``@cb.on_end`` (after the event loop). Static hooks run bottom-up.

   Predicate
      A ``@cb.predicate`` method returning a bool, used with
      ``Condition.wait_until``.

   Function
      A ``@cb.function`` method with annotated parameters, callable from
      compiled code like a method. Subclasses may override it; the model's
      actual class decides which implementation runs.

   Reference sweep
      A sweep of a ``Ref`` field over models in the tree, running each
      choice as its own design point. Used to compare policies.

   Event
      A ``@cb.event`` method run once at a time scheduled with
      ``cb.schedule``.

   Signal
      (1) The integer a blocking call returns to explain why the process
      resumed: ``0`` success, ``-1`` preempted, ``-2`` interrupted, positive
      values user-defined. (2) :class:`cimba.Signal`, the result object for
      a captured entity.

   Input
      ``cb.Input[T]``: a sequence of real-world values consumed with
      ``next()``. Its source is configured, not coded.

   Series
      ``cb.Series[T]``: time-indexed real-world values in fixed-width
      buckets, read with ``now()`` or ``at(t)``.

   Source
      What feeds an input: a distribution (``inputs.dist``), a recorded trace
      (``inputs.trace``), a bootstrap resample (``inputs.bootstrap``) or a
      fitted time-series model (``inputs.fitted``). Immutable, with
      provenance.

   Input stream
      The independent random stream one input uses in one trial, derived
      from the trial seed and the input's path or tag.

   Exhaustion policy
      What happens when a finite source runs out: ``fail``, ``wrap``,
      ``end_trial`` or ``extend``.

   Prefix stability
      The property that a source generating *2n* values from a stream yields
      the same first *n* values as generating *n*. It makes transparent
      extension exact.

   Provenance
      The ``describe()`` record of which source produced an input's values:
      method, parameters and data fingerprints. Stored in the results.

   Sweep
      A design axis created with ``cb.sweep`` (independent) or ``cb.sweeps``
      (linked), assigned to a ``Param``, a distribution parameter, an input
      or a ``Ref``.

   Design point
      One combination of sweep values.

   Replication
      One independent repetition of a design point, with its own seed.

   Trial
      One design point × one replication: a complete, independent
      simulation run and the unit of parallel work.

   Window
      :class:`cimba.Window`: warmup, measurement duration and cooldown, or
      run-until-idle.

   Common random numbers
      CRN. The default seeding, where replication *r* uses the same seed at
      every design point, so comparisons between points are paired.

   Capture
      Keeping the raw history of an entity, enabled with ``entity.capture()``
      before the run.

   Samples
      :class:`cimba.Samples`: one output across the design, a read-only
      ``(points, replications)`` array.

   Results
      :class:`cimba.Results`: the immutable outcome of a run, indexed by
      model object.

   Canonical path
      A label such as ``network.facilities[2].demand`` identifying an
      instance or field. Used in messages, descriptions, native names and
      stream identity; never used for addressing.

   Host
      The ordinary Python process where you configure models and analyze
      results, as opposed to the compiled trial code.
