Models, Components and Experiments
==================================

Model declarations
------------------

Use :mod:`cimba.sim` for modeling. A model is a :class:`~cimba.sim.Model`
subclass whose annotated fields are typed by their simulation role:

``Model``, ``Component``, ``Experiment``, ``Env``, ``Handle``, ``Param``,
``Output``, ``State``, ``FloatState``, ``Const``, ``Queue``, ``Resource``,
``Pool``, ``Store``, ``Dataset``, ``Condition``, ``Predicate``, ``Event``,
``Processes``, ``PQueues``, ``Ref``, ``Refs``, ``Struct``,
``Trace``, ``capacity()``, ``count()``, ``process()``, ``predicate()``,
``event()``, ``collect()``, ``function()``.

``sim.Param`` values are expanded into parameter combinations, ``sim.Output``
values are collected after each trial, ``sim.State`` and ``sim.FloatState`` hold
mutable trial-local state, and entities such as ``sim.Queue``, ``sim.Resource``,
``sim.Pool``, ``sim.Store``, ``sim.PQueues``, ``sim.Condition``, and
``sim.Dataset`` are created for each trial.

A parameter may declare a scalar default with normal Python syntax,
``mean_service: sim.Param = 0.25``. Omitted experiment arguments use that
value; explicit scalar or swept values override it. Parameters without defaults
remain required, and ``model.param_defaults`` reports defaults by flattened
field name.

.. code-block:: python

   import cimba.sim as sim

   class Clinic(sim.Model):
       arrival_rate: sim.Param
       wait_time: sim.Output
       queue: sim.Queue
       doctor: sim.Resource
       waits: sim.Dataset

   model = Clinic("clinic")

Breaking change: callbacks live on the class
--------------------------------------------

Model callbacks are class declarations. The former instance-bound decorators
have been removed without compatibility aliases. Move each callback into the
``sim.Model`` subclass and use the exported marker:

.. code-block:: python

   # Before (no longer supported)
   model = Clinic("clinic")

   @model.process
   def arrivals(env: Clinic):
       ...

.. code-block:: python

   # Now
   class Clinic(sim.Model):
       @sim.process
       def arrivals(env: "Clinic"):
           ...

   model = Clinic("clinic")

The same change applies to ``collect``, ``predicate``, and ``event``. If a
callback publishes into a declared ``sim.Processes``, ``sim.Predicate``, or
``sim.Event`` field, give the callback a distinct method name and bind it with
``field="field_name"``. Behavioral variants should be subclasses; direct
``sim.Model(...)`` construction is supported only for callback-free models.

Components
----------

Components group related declarations and process methods. Methods decorated
with top-level ``@sim.process`` are lowered into ordinary model processes at
model construction, and model callbacks can read component fields with
``env.retailer.orders``. Component fields are exposed in experiments with
flattened names such as ``retailer__orders``. Methods decorated with
top-level ``@sim.collect`` run once per instance at the end of each trial,
before the model-level ``@sim.collect`` callback, typically assigning the
component's ``sim.Output`` fields.

Read-only synchronous behavior is declared with top-level ``@sim.function``.
Its non-``self`` parameters and return value must be explicitly annotated as
``bool``, ``int``/``sim.Handle``, or ``float``. A process or collector can call
``env.policy.decide(level)``, and another method on the same component can call
``self.decide(level)``. The helper may read scalar component parameters,
outputs, state, and explicitly declared ``sim.Const`` values, including through
nested components and ``Ref``/``Refs`` paths, but cannot mutate fields or call
scheduling and entity operations.

Components may contain other components, and flattened names follow the same
recursive convention, for example ``env.attraction.queues.line`` becomes
``attraction__queues__line``. Nested component process methods are also lowered
with their component path in the process name.

Components may declare a spawnable process with
``@sim.process(spawnable=True)``. It can be
spawned from component or model code with natural paths such as
``sim.spawn(self.visitor, env)`` or
``sim.spawn(env.park.entrance.visitor, env)``. These component processes
may receive a final ``sim.Struct`` view parameter.

Components may reference other declared components with ``sim.Ref[Target]``
fields and routing tables of collection items with ``sim.Refs[Target]``,
letting method bodies route through paths such as
``self.downstream.inbox.put(h)`` or
``self.routes[i].inbox.put(h)``; see
:doc:`../advanced/components` for wiring and routing details.

Fixed repeated structures can be declared with standard ``list[Component]``
annotations, for example ``attractions: list[Attraction] = [...]``. Model
callbacks can use indexed access such as ``env.attractions[i].queues[j]``;
runtime fields remain flattened, for example ``attractions__queues``. Nested
collections are linearized behind the scenes, so
``env.campus.zones[i].gates[j].queue`` remains valid model source while the
trial table stores a one-dimensional ``campus__zones__gates__queue`` field.

Per-process fields
------------------

Declare a ``sim.Struct`` subclass with ``float`` and ``int`` annotations. A
process can receive its own field view as a final annotated parameter:
``def visitor(env, view: Visitor)``. Multi-copy processes can also receive the
copy index: ``def visitor(env, idx, view: Visitor)``. ``Visitor(handle)``
returns a read/write view of another process's fields when model code already
has that process handle.

Compilation plans and cache
---------------------------

Reusable class-declared model and component callbacks are planned from the
first normally constructed model instance; importing a module or defining a
model class does not construct
a hidden prototype. ``Model.compilation_status()`` reports ``pending``,
``ready``, ``failed``, or ``unavailable`` together with elapsed time, callback
counts, persistent-cache hits/misses, and an error message when preparation
failed. ``Model.compilation_plan()`` returns the immutable
``sim.CompilationPlan`` after a plan has been built.
After an instance compiles its remaining processes, predicates, events, and
collectors, ``model.callback_cache_stats()`` reports their cache hits, misses,
and writes separately from the reusable class preparation.

The default ``__cimba_precompile__ = "eager"`` prepares reusable class
callbacks during the first real model construction. A subclass can select
``"lazy"`` to prepare them on its first experiment or ``"explicit"`` and call
``Model.precompile(*constructor_args, **constructor_kwargs)`` itself. Explicit
precompilation retries a previous failure, which is useful when callback
globals are initialized later during module startup.

Compiled native callbacks are cached by code, signature, record layout,
compiler versions, operating system, architecture, and CPU target. The cache
is enabled by default. Set ``CIMBA_CACHE=0`` to disable both memory and disk
reuse, or ``CIMBA_CACHE_DIR`` to choose the persistent cache directory. Cache
entries are optimization-only: an absent, stale, or unreadable entry falls
back to normal compilation. Process-local handles returned by
``sim.log_text()`` are placed in a runtime sidecar, so callbacks that write
logs or reports can safely reuse persisted object code in another process.

Process graphs
--------------

Call ``model.process_dag()`` to infer a resource-aware graph from class-declared
process bodies. The returned ``ProcessDAG`` contains ``ProcessDAGNode`` and
``ProcessDAGEdge`` records for processes and model fields, and can render
Mermaid or Graphviz DOT text. The inference follows direct ``sim`` calls,
simple aliases, helper functions called with ``env``, spawnables, stores,
priority queues, conditions, events, mutable state, and shared resources.
Synchronous component methods appear as ``function:`` nodes, with ``read``
edges from referenced parameters/state and ``call`` edges from processes or
other functions:

.. code-block:: python

   graph = model.process_dag()
   print(graph.to_mermaid())
   print(graph.to_dot())

Experiments
-----------

``model.experiment(...)`` returns an ``Experiment``; ``exp.run()`` executes
the trial table in place and returns the number of failed trials, and
``exp["field"]`` reads any trial column as an array. ``exp.summary()``
condenses the outputs across replications: it returns a structured array with
one record per design point holding the swept parameter values and, for each
output, its replication mean (``name``) and Student-t confidence-interval
half-width (``name_hw``, 95% by default)::

   exp = model.experiment(utilization=[0.7, 0.8, 0.9], replications=20,
                          duration=10_000.0, seed=42)
   exp.run()
   for row in exp.summary("avg_wait"):
       print(f"rho={row['utilization']:.1f}  "
             f"wait={row['avg_wait']:.2f} +- {row['avg_wait_hw']:.2f}")

``exp.summary("a", "b", confidence=0.99)`` selects outputs and the confidence
level; failed trials (NaN) are excluded per output. ``exp.replications`` and
``exp.swept`` expose the layout (trial order is design-point-major with
replications innermost).

Typed result namespaces
~~~~~~~~~~~~~~~~~~~~~~~

Experiments also expose retained structured results through ``exp.results``.
Output paths follow the model's component structure, so callers do not need to
construct flattened ``__`` names:

.. code-block:: python

   exp.run()
   queue_means = exp.results.counters.mean_queue_length
   served = exp.results.customers_served

Output leaves are the same NumPy trial-column views returned by
``exp["..."]``; dtype, replication order, and component collection axes are
unchanged. A component collection therefore remains one array with its
collection dimension, rather than becoming one attribute per item.

Outputs, captured datasets, and captured histories share the model's object
tree. Callers do not need to know which storage mechanism produced a result,
and a result declared inside a component is found through that component:

.. code-block:: python

   all_wait_samples = exp.results.waits
   all_station_samples = exp.results.station.samples
   all_queue_rows = exp.results.station.queue

Dataset leaves match ``exp.datasets(name)`` and history leaves match
``exp.histories(name)``: both are tuples aligned with experiment trials, with
an additional inner tuple for indexed component histories. The existing
``dataset()``, ``datasets()``, ``history()``, and ``histories()`` methods keep
their original behavior and error handling. Output views share the trial
table's storage; captured datasets and histories are the copied arrays already
returned by their existing methods.

The runtime namespaces discover declared/captured names through ``dir()`` and
raise a path-aware ``AttributeError`` for unknown names. Dynamically named
outputs remain available through the string-key API, which is the general
fallback for names not represented by the namespace.

For exact model-specific static completion, parameterize ``Model`` with a
result ``Protocol`` describing this shared object tree. Pyright then propagates
that schema to ``model.experiment().results``::

   class QueueResults(Protocol):
       customers_served: NDArray[np.float64]
       waits: tuple[NDArray[np.float64], ...]
       counters: CounterResults

   class QueueModel(sim.Model[QueueResults]):
       ...

Unparameterized models continue to use the general dynamic result namespace.

If a model-level collector declares ``env.<entity>.history().capture()``,
``exp.history("field", trial=i)`` returns that trial's raw time-series rows as
a NumPy array with columns ``time``, ``value``, and ``duration``.
``exp.histories("field")`` returns one such array per trial, aligned with the
experiment row order. For fields owned by a component collection, indexed
captures return one inner array per collection item; use
``exp.history("field", trial=i, index=j)`` to select one item.

If a model-level collector declares ``env.<dataset>.capture()``,
``exp.dataset("field", trial=i)`` returns that trial's raw dataset samples as a
one-dimensional NumPy array. ``exp.datasets("field")`` returns one array per
trial, also aligned with the experiment row order.
