Process Graphs
==============

``model.process_dag()`` infers a graph from registered process source code. It
is a modeling aid: useful for review, documentation, and spotting unexpected
dependencies before a model becomes hard to reason about.

Rendering a graph
-----------------

.. code-block:: python

   graph = model.process_dag()
   print(graph.to_mermaid())
   print(graph.to_dot())

Mermaid output works well in Markdown-oriented docs. DOT output can be rendered
with Graphviz tooling. The graph records are also available directly through
``graph.nodes`` and ``graph.edges``.

What the graph means
--------------------

Nodes represent processes and model fields. Edges represent visible
interactions in the process source:

* putting into or getting from queues and stores,
* acquiring resources and pools,
* waiting on or signaling conditions,
* scheduling and waiting on events,
* spawning dynamic processes,
* reading or writing ``sim.State`` and ``sim.FloatState`` fields,
* direct process-handle operations such as ``sim.interrupt()``.

For example, a clinic model may show an arrival process feeding a waiting-room
queue, a clerk process consuming that queue, and a collector reading state.

Review workflow
---------------

Process graphs are most useful when they are part of model review:

.. code-block:: python

   graph = model.process_dag()
   mermaid = graph.to_mermaid()
   assert "waiting_room" in mermaid

Use the graph to ask modeling questions:

* Which processes create work?
* Which shared entities are coordination points?
* Are any state fields written from more places than expected?
* Are dynamic processes spawned only by the intended process?
* Does the model contain feedback loops that should be documented?

Cycles and limitations
----------------------

The name says DAG, but many real simulation models contain intentional cycles.
A condition may wake a process, and that process may later signal the same
condition. Such graphs can still render. ``graph.topological_order()`` raises
``ValueError`` when a strict acyclic order does not exist.

Inference is static. It reads Python source; it does not run the model. It can
follow common direct calls, simple aliases, and some helper functions, but it
cannot prove every dynamic handle flow through arbitrary Python structures.

Treat the graph as a map for human review, not as a proof of correctness.

For the ``ProcessDAG`` API, see :doc:`../api_reference/models`.
