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
   def arrivals(self: Clinic):
       ...

.. code-block:: python

   # Now
   class Clinic(sim.Model):
       @sim.process
       def arrivals(self: "Clinic"):
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
model construction, and model callbacks use ``self`` as the root trial
environment; they can read component fields with ``self.retailer.orders``.
Component fields are exposed in experiments with
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

The root model may own the same kind of helper. Its first argument is ``self``,
which represents the root trial environment view. Model callbacks call it
through ``self.<name>(...)``; component callbacks call it through
``env.<name>(...)``:

.. code-block:: python

   class Inventory(sim.Model):
       stock: sim.State

       @sim.function
       def shortage(self: "Inventory", demand: float) -> float:
           return max(0.0, demand - float(self.stock))

Root helpers may read scalar model fields and component scalar namespaces and
may call other root or component functions. They follow the same annotation,
read-only, and non-recursion rules as component functions.

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
They may also pass ``struct=SomeStruct`` to attach storage without injecting a
view; if both forms are used they must name the same struct type.

Components may reference other declared components with ``sim.Ref[Target]``
fields and routing tables of collection items with ``sim.Refs[Target]``,
letting method bodies route through paths such as
``self.downstream.inbox.put(h)`` or
``self.routes[i].inbox.put(h)``; see
:doc:`../advanced/components` for wiring and routing details.

Fixed repeated structures can be declared with standard ``list[Component]``
annotations, for example ``attractions: list[Attraction] = [...]``. Model
callbacks can use indexed access such as ``self.attractions[i].queues[j]``;
runtime fields remain flattened, for example ``attractions__queues``. Nested
collections are linearized behind the scenes, so
``self.campus.zones[i].gates[j].queue`` remains valid model source while the
trial table stores a one-dimensional ``campus__zones__gates__queue`` field.

Per-process fields
------------------

Declare a ``sim.Struct`` subclass with ``float`` and ``int`` annotations. A
process can receive its own field view as a final annotated parameter:
``def visitor(self, view: Visitor)``. Multi-copy processes can also receive the
copy index: ``def visitor(self, idx, view: Visitor)``. ``Visitor(handle)``
returns a read/write view of another process's fields when model code already
has that process handle.

Compilation and reuse
---------------------

Construction collects and lowers declarations. ``model.compile()`` compiles a
fully constructed model and returns it. ``model.experiment()`` calls the same
compilation step automatically. Compilation errors propagate to the caller;
there is no hidden preparation attempt or class-wide fallback.

Reuse the same model for experiments with different parameters, traces, seeds,
and replication counts. Native user callbacks belong to that model. Construct
a new model after changing callback code or compile-time configuration. Numba
captures globals during compilation; values that should vary between
experiments belong in ``sim.Param`` or ``sim.Trace``.

Only Cimba's fixed lifecycle callbacks are shared between models. User code is
not loaded from a persistent object-code cache. The previous
``__cimba_precompile__`` modes, class ``precompile()``, ``compilation_plan()``,
``compilation_status()``, ``callback_cache_stats()``, and their result types
have been removed. Replace them with an ordinary instance:

.. code-block:: python

   model = Network(...).compile()
   first = model.experiment(...)
   second = model.experiment(...)

``CIMBA_CACHE`` and ``CIMBA_CACHE_DIR`` no longer affect compilation. Old cache
files are unused and can be removed. For timing measurements, use
``benchmark/component_compilation.py``; its version 2 JSON separates explicit
compilation, experiment construction, and execution.

Synchronous helpers receive the trial record and read fields inside their own
control flow. Arguments and dynamic receivers are evaluated once. Locally
computed indices can access ``Const`` tables as well as stored scalar fields.
Native entity/statistics methods accept keyword arguments, but reject
out-of-order keyword expressions that could change evaluation order when
converted to native positional arguments. Evaluate those expressions into
local variables first, or supply them in parameter order.

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
level; failed trials are excluded from every output, and missing NaN values
from successful trials are excluded per output. ``exp.replications`` and
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

If a model-level collector declares ``self.<entity>.history().capture()``,
``exp.history("field", trial=i)`` returns that trial's raw time-series rows as
a NumPy array with columns ``time``, ``value``, and ``duration``.
``exp.histories("field")`` returns one such array per trial, aligned with the
experiment row order. For fields owned by a component collection, indexed
captures return one inner array per collection item; use
``exp.history("field", trial=i, index=j)`` to select one item.

If a model-level collector declares ``self.<dataset>.capture()``,
``exp.dataset("field", trial=i)`` returns that trial's raw dataset samples as a
one-dimensional NumPy array. ``exp.datasets("field")`` returns one array per
trial, also aligned with the experiment row order.

Trial outcomes and replay
-------------------------

Every ``exp.run()`` starts ``State`` and ``FloatState`` fields at zero and
outputs at NaN, retaining experiment parameters, traces, and seeds. Repeating
a run therefore repeats the same experiment for deterministic callbacks.
Use a parameter and assign it in a process when you need nonzero initial
state; writing state directly into ``exp.trials`` before ``run()`` is no longer
an initialization mechanism.

``exp.failures`` and the return value of ``run()`` are the native failure count.
``exp.failed`` is a Boolean array identifying failed trials. Both native trial
abandonment and uncaught compiled callback exceptions count as failures, and
all scalar outputs of those trials become NaN. A successful trial may itself
produce NaN, or declare no outputs, without being classified as failed.
Captured datasets and histories can contain partial data from a failed trial;
filter them using ``exp.failed`` before analyzing successful trials.

``sim.Trace(field)`` exposes read-only replay storage inside compiled code.
Call ``.copy()`` if a callback needs a mutable working array. The Python arrays
supplying an experiment must remain unchanged while it runs. Trial tables
contain native pointers and are runtime views, not a serialization format.

``cimba.use_threads(n)`` now sets the native worker count for subsequent runs;
zero selects the native CPU-count default. Call it between runs. Calls to
``run()`` on the same experiment are serialized, as is compilation of the same
model from multiple Python threads.
