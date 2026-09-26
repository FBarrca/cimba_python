Modeling
========

Every part of the model is a ``cimba.Model`` class. A model may contain child
models or references to other models. Annotated fields declare trial data:

* ``Param[T]`` is an experiment parameter; ``State[T]`` is mutable trial
  state; ``Output[T]`` is measured at the end of the trial.
* ``Input[T]`` and ``Series[T]`` declare real-world variability. Source
  assignment happens on the Python object before running.
* ``Ref[M]`` and ``list[M]`` connect model instances.
* ``Container``, ``Store[T]``, ``PriorityStore[T]``, ``Resource``,
  ``Condition``, and ``Dataset`` are native simulation entities.

Mark concurrent methods with ``@cimba.process``. ``@cimba.on_start`` and
``@cimba.on_end`` run before and after the event loop, with children before
parents. ``cimba.hold()``, ``cimba.now()``, entity methods, spawning, and
scheduling work inside compiled methods. Process methods support the Numba
nopython subset. Constructing a model only configures Python objects;
compilation starts when the experiment runs.

A spawned model can receive an input or series handle from its parent in a
``spawn(..., input_field=self.input_field)`` call. The spawned field gets an
independent cursor and deterministic distribution stream. Its input slot is
checked and released with the spawned model at trial cleanup.

See ``tutorial/tut_3_1.py`` for dynamic visitors, ``tutorial/tut_5_1.py`` for
typed store handoffs, and ``tutorial/multi_echelon_inventory.py`` for a
heterogeneous collection with shared references.
