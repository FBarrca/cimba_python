Process Graphs
==============

``sim.Model`` can infer a resource-aware process graph from registered process
functions. This is useful while building or reviewing a model: it shows which
processes feed queues, stores, conditions, events, mutable state, spawnable
processes, and other simulation entities.

The graph is inferred from Python source code. You do not annotate edges by
hand.

.. code-block:: python

   import cimba.sim as sim


   class MM1(sim.Model):
       queue: sim.Queue


   model = MM1()


   @model.process
   def arrival(env: MM1):
       sim.put(env.queue, 1)


   @model.process
   def service(env: MM1):
       sim.get(env.queue, 1)


   graph = model.process_dag()
   print(graph.to_mermaid())

For this model, Mermaid output is:

.. code-block:: text

   flowchart TD
       n_process_arrival["arrival"]
       n_process_service["service"]
       n_queue_queue[("queue")]
       n_process_arrival -->|put| n_queue_queue
       n_queue_queue -->|get| n_process_service

What Is Inferred
----------------

The graph contains process nodes and model-field nodes. Process nodes use the
registered process names. Model-field nodes use the field names declared as
``sim.Queue``, ``sim.Store``, ``sim.PQueues``, ``sim.Condition``,
``sim.Resource``, ``sim.Pool``, ``sim.Event``, ``sim.State``, or
``sim.FloatState``.

The inference recognizes common process interactions:

* ``sim.put()``, ``sim.store_put()``, ``sim.pq_put()``, and ``sim.signal()``
  create edges from a process to a resource.
* ``sim.get()``, ``sim.store_get()``, ``sim.store_take()``, ``sim.pq_get()``,
  ``sim.pq_take()``, and ``sim.wait_for()`` create edges from a resource to a
  process.
* ``sim.spawn(env.<spawnable>, env)`` creates an edge from the spawning process
  to the spawned process.
* ``sim.schedule()`` and ``sim.schedule_at()`` create edges from the scheduling
  process to a ``sim.Event`` node. ``sim.wait_event()`` creates an edge from the
  event node to the waiting process when the event handle can be traced back to
  the schedule call.
* Event callback bodies registered with ``@model.event`` are inspected for
  mutable state interactions.
* Assignments to ``sim.State`` and ``sim.FloatState`` fields create ``write``
  edges from the process or event callback. Reads from those fields create
  ``read`` edges into the process or event callback.
* Direct process-handle operations such as ``sim.interrupt(env.worker[0], ...)``
  create process-to-process edges when the target can be resolved from a
  ``sim.Processes`` field.
* Shared capacity interactions such as ``sim.pool_acquire()`` and
  ``sim.acquire()`` are shown as ``uses`` edges from the process to the pool or
  resource. They do not create arbitrary process-to-process dependencies.

Simple local aliases are tracked:

.. code-block:: python

   @model.process
   def service(env: MM1):
       q = env.queue
       sim.get(q, 1)

The analyzer also follows simple helper functions called with ``env``, including
helpers decorated with ``@numba.njit`` when their original Python function is
available.

Rendering
---------

``ProcessDAG`` can render to Mermaid or Graphviz DOT text:

.. code-block:: python

   graph = model.process_dag()
   mermaid = graph.to_mermaid()
   dot = graph.to_dot()

Mermaid output is convenient for Markdown and many documentation systems. DOT
output can be passed to Graphviz tools.

``graph.nodes`` contains ``ProcessDAGNode`` records. Each node has a stable
``key`` such as ``"process:arrival"`` or ``"queue:queue"``. ``graph.edges``
contains ``ProcessDAGEdge`` records whose ``source`` and ``target`` are those
keys.

Acyclic Order
-------------

``graph.topological_order()`` returns node keys in topological order when the
inferred graph is acyclic:

.. code-block:: python

   print(graph.topological_order())

Some models intentionally contain feedback. For example, a condition may wake a
process, and that process may later signal the same condition after releasing
resources. Such graphs still render, but ``topological_order()`` raises
``ValueError`` because the graph is not a strict DAG.

Limitations
-----------

Inference is static. It inspects source code; it does not run a simulation and
does not observe every runtime branch. For best results, define models and
process functions in regular Python modules. Functions created interactively or
from stdin may not have source code available to ``inspect.getsource()``.

The graph is a modeling aid rather than a proof of all possible runtime
interactions. Dynamic handle flows through arbitrary Python data structures,
deep reflection, or heavily indirect helper dispatch may not be inferred.
