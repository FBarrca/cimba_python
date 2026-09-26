Draw your model
===============

**Goal:** see how a model is put together and how its processes interact,
before (or instead of) reading all of its code.

``cimba.diagrams`` draws two complementary pictures of a configured model.
Neither compiles nor runs anything, and both work before you've bound input
data, so you can draw a model while you're still building it.

.. list-table::
   :widths: 30 70

   * - :func:`~cimba.diagrams.structure`
     - The **object tree**: which model owns which (solid arrows) and which
       references which (dotted arrows).
   * - :func:`~cimba.diagrams.process_graph`
     - The **interactions**: which process puts into or gets from which
       store, acquires which resource, reads which input, signals which
       condition and spawns which model.

Both return a :class:`~cimba.diagrams.Graph` that renders to
`Mermaid <https://mermaid.js.org/>`_ or `Graphviz <https://graphviz.org/>`_
DOT.

Process graphs
--------------

.. code-block:: python

   from cimba import diagrams
   from tutorial.tut_1_1 import MM1

   graph = diagrams.process_graph(MM1())
   print(graph.to_mermaid())

.. code-block:: text

   flowchart TD
     subgraph g0["mm1: MM1"]
       n0[/"interarrival · exponential"/]
       n1[/"service_time · exponential"/]
       n2["queue · Container"]
       n3(["arrival"])
       n4(["service"])
     end
     n0 -->|next| n3
     n3 -->|put| n2
     n2 -->|get| n4
     n1 -->|next| n4

Rendered:

.. cimba-diagram:: tutorial.tut_1_1:MM1
   :direction: LR

How to read it:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Shape
     - Meaning
   * - rounded box
     - a process (``×3`` when it runs in several copies)
   * - rectangle
     - an entity, with its type: ``queue · Container``
   * - parallelogram
     - an input, with its bound source: ``interarrival · exponential``
       (``no source`` if none is bound yet, ``sweep of 3`` for a sweep)
   * - hexagon
     - an ``@cb.event`` method
   * - double-bordered box
     - ``new Visitor``: where a spawned model enters
   * - boxes around nodes
     - one per model instance (``harbor.facilities: HarborFacilities``),
       plus one per spawned model class (``Ship (spawned)``)

Edges point the way things flow. A process **puts** into a store; the store
feeds the process that **gets** from it; a resource feeds the process that
**acquires** it, and the process **releases** it back; an input feeds the
process that calls **next**/**now**/**at**; a condition feeds the process that
**waits**, and a process **signals** it; a process **spawns** a model and
**schedules** an event; any process **records** into a dataset.

A bigger example
~~~~~~~~~~~~~~~~

The harbor from tutorial chapter 4: ships are spawned by the traffic model
and, through their ``harbor`` reference, compete for the facilities.

.. cimba-diagram:: tutorial.tut_4_1:Harbor
   :direction: LR

Options
~~~~~~~

``process_graph(model, state=True)``
   Also draw writes to ``Param``/``State``/``Output`` fields, as cylinders.
   Useful to spot state written from more places than expected.

``process_graph(model, hooks=True)``
   Include ``on_start``/``on_end`` hooks as actors.

``graph.to_mermaid(direction="LR")`` / ``graph.to_dot(rankdir="LR")``
   Choose the layout direction.

The structure graph
-------------------

.. code-block:: python

   print(diagrams.structure(model).to_mermaid())
   # diagrams.mermaid(model) is a shortcut for the same text

.. cimba-diagram:: tutorial.tut_5_1:AssemblyLine
   :kind: structure
   :direction: LR

Rendering
---------

* **Documentation and Markdown.** GitHub, GitLab, Notion, Obsidian and many
  wikis render fenced code blocks marked ``mermaid`` directly. Sphinx sites can
  use `sphinxcontrib-mermaid <https://github.com/mgaitan/sphinxcontrib-mermaid>`_
  (these docs do).
* **Notebooks.** Paste into `mermaid.live <https://mermaid.live>`_, or
  render DOT with the ``graphviz`` Python package:
  ``graphviz.Source(graph.to_dot())``.
* **Files.** ``Path("model.mmd").write_text(graph.to_mermaid())``, or
  ``dot -Tsvg model.dot -o model.svg`` for Graphviz.

Working with the graph
----------------------

The graph is plain data: ``graph.nodes`` (each with ``key``, ``label``,
``kind`` and ``group``), ``graph.edges`` (``source``, ``target``, ``label``,
``style``) and ``graph.groups``. Keys are canonical paths such as
``harbor.facilities.tugs``, or ``Ship.voyage`` for a spawned model's process.
That makes graphs easy to check in tests:

.. code-block:: python

   graph = diagrams.process_graph(Harbor())
   edges = {(e.source, e.label, e.target) for e in graph.edges}
   assert ("harbor.facilities.tugs", "acquire", "Ship.voyage") in edges

``graph.topological_order()`` returns node keys in flow order, or raises
``ValueError`` if the graph has a cycle. Many real models do: a process that
signals a condition it also waits on is an intentional cycle.

Review questions
----------------

A process graph is a good agenda for a model review:

* Which processes create work, and which consume it? Is every store both
  filled and emptied?
* Which entities are coordination points shared by several processes?
* Does every condition that someone waits on also get signalled?
* Are spawned models connected to the static models they need, and are they
  released somewhere?
* Is every input read by the process you expect, and bound to the source you
  expect?

How it works, and its limits
----------------------------

The graph is inferred from the **source code** of each process, event and
(optionally) hook, resolved against the **configured objects**. Attribute
chains like ``self.harbor.facilities.tugs`` are followed through children,
references and list items. The analysis also follows:

* local aliases (``berths = facilities.berths_small``);
* list indices: a literal index hits one item, a computed index hits every
  item that could be meant;
* helper functions called with the model (plain or ``numba.njit``);
* the keyword arguments of ``cb.spawn``, which connect a spawned model's
  references and inputs to the objects the spawner passed in;
* models taken out of typed stores (``part = self.inbox.get()``).

It is a map for humans, not a proof. It can't see through process handles
stored as integers (``cb.Process(pointer).interrupt()``), values computed at
run time, or code whose source isn't available. When in doubt, it leaves an
edge out rather than inventing one.
