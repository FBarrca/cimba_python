Models, Components and Experiments
==================================

Model declarations
------------------

Use :mod:`cimba.sim` for modeling. A model is a :class:`~cimba.sim.Model`
subclass whose annotated fields are typed by their simulation role:

``Model``, ``Component``, ``Experiment``, ``Env``, ``Handle``, ``Param``,
``Output``, ``State``, ``FloatState``, ``Queue``, ``Resource``, ``Pool``,
``Store``, ``Dataset``, ``Condition``, ``Predicate``, ``Event``,
``Processes``, ``PQueues``, ``Spawnable``, ``Struct``, ``Trace``,
``capacity()``, ``count()``, ``process()``, ``collect()``.

``sim.Param`` values are expanded into parameter combinations, ``sim.Output``
values are collected after each trial, ``sim.State`` and ``sim.FloatState`` hold
mutable trial-local state, and entities such as ``sim.Queue``, ``sim.Resource``,
``sim.Pool``, ``sim.Store``, ``sim.PQueues``, ``sim.Condition``, and
``sim.Dataset`` are created for each trial.

.. code-block:: python

   import cimba.sim as sim

   class Clinic(sim.Model):
       arrival_rate: sim.Param
       wait_time: sim.Output
       queue: sim.Queue
       doctor: sim.Resource
       waits: sim.Dataset

   model = Clinic("clinic")

Components
----------

Components group related declarations and process methods. Methods decorated
with top-level ``@sim.process`` are lowered into ordinary model processes at
model construction, and model callbacks can read component fields with
``env.retailer.orders``. Component fields are exposed in experiments with
flattened names such as ``retailer__orders``. Methods decorated with
top-level ``@sim.collect`` run once per instance at the end of each trial,
before the model-level ``@model.collect`` callback, typically assigning the
component's ``sim.Output`` fields.

Components may contain other components, and flattened names follow the same
recursive convention, for example ``env.attraction.queues.line`` becomes
``attraction__queues__line``. Nested component process methods are also lowered
with their component path in the process name.

Components may declare ``sim.Spawnable`` fields. A component-owned spawnable
binds to a same-named ``@sim.process`` method on that component, and can be
spawned from component or model code with natural paths such as
``sim.spawn(self.visitor, env)`` or
``sim.spawn(env.park.entrance.visitor, env)``. Spawnable component processes
may receive a final ``sim.Struct`` view parameter.

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

Process graphs
--------------

Call ``model.process_dag()`` to infer a resource-aware graph from registered
process bodies. The returned ``ProcessDAG`` contains ``ProcessDAGNode`` and
``ProcessDAGEdge`` records for processes and model fields, and can render
Mermaid or Graphviz DOT text. The inference follows direct ``sim`` calls,
simple aliases, helper functions called with ``env``, spawnables, stores,
priority queues, conditions, events, mutable state, and shared resources:

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
