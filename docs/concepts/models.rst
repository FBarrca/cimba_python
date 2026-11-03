Models and fields
=================

A model is a subclass of :class:`cimba.Model`. Its **annotated fields**
declare what every instance has; its **decorated methods** declare what it
does. There is one concept for every "thing" in a simulation, whether it's the
whole system, a station in it, or a customer passing through.

.. code-block:: python

   class Station(cb.Model):
       service_time: cb.Input[float] = inputs.dist.exponential(mean=4.0)  # input
       speed: cb.Param[float] = 1.0                                        # parameter
       busy_since: float = 0.0                                             # plain field
       served: cb.Output[int]                                              # output
       queue: cb.Store[Job]                                                # entity
       worker: cb.Resource = cb.Resource(capacity=2)                       # configured entity
       plant: cb.Ref["Plant"]                                              # reference
       next_station: cb.Ref["Station"] | None = None                       # optional reference
       arrivals: cb.Array[int] = cb.Array(4)                               # array

The kinds of field
------------------

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Declaration
     - Kind
     - What it is
   * - ``x: cb.Param[T]``
     - parameter
     - A value fixed for a whole trial. Set it on the class, in ``__init__``
       or on the instance. It may be a ``cb.sweep``. Readable (and, rarely,
       writable) in compiled code.
   * - ``x: T = v``
     - plain field
     - A per-trial value. It starts at ``v`` in every trial and compiled code
       may change it. A static model needs a value, on the class or set in
       ``__init__``.
   * - ``x: cb.Output[T]``
     - output
     - A per-trial value that is also a result: its value at the end of the
       trial is collected into :class:`~cimba.Samples`. It starts at its
       default, if it has one; otherwise at ``nan`` (``float``) or ``0``
       (``int``, ``bool``).
   * - ``x: cb.Input[T]``
     - input
     - A sequence of real-world values, consumed with ``next()``. Its source
       is configured, not coded. See :doc:`inputs`.
   * - ``x: cb.Series[T] = cb.Series(step=, origin=)``
     - series
     - Time-indexed real-world values, read with ``now()`` or ``at(t)``.
   * - ``x: cb.Container`` (etc.)
     - entity
     - A native queue, store, resource, condition or dataset. See
       :doc:`entities`.
   * - ``x: OtherModel``
     - child
     - A model owned by this one: part of the tree. It can be swept over
       several models: one tree per option.
   * - ``x: list[OtherModel]``
     - collection
     - A fixed-size list of owned child models; items may be subclasses.
   * - ``x: cb.Ref[OtherModel]``
     - reference
     - A pointer to another model somewhere in the tree. Add ``| None`` to
       make it optional.
   * - ``x: list[cb.Ref[OtherModel]]``
     - collection of references
     - A list of pointers, not ownership. Items may be ``None``.
   * - ``x: cb.Array[T] = cb.Array(n)``
     - array
     - A fixed-length table of ``int``, ``float`` or ``bool`` per instance,
       whose length is configuration. See :ref:`array-fields`.
   * - (no annotation)
     - host-only
     - An ordinary Python attribute, such as a display name. Compiled code
       never sees it.

Scalar field types (``T``) are ``float``, ``int``, ``bool`` and
``enum.IntEnum`` classes (see below). A table of numbers that model code
changes is a ``cb.Array`` field. Strings and other Python objects can't live
in trial memory. Keep them as host-only attributes, or as module-level
constants (NumPy arrays work well) that compiled code reads.

Enum fields
~~~~~~~~~~~

A ``Param``, plain field or ``Output`` may be typed with an ``enum.IntEnum``.
Trial memory holds the member's integer value; everywhere else the field
holds members:

.. code-block:: python

   class NodeKind(enum.IntEnum):
       START = 0
       TASK = 2
       EXCLUSIVE = 3

   class Node(cb.Model):
       kind: cb.Param[NodeKind] = NodeKind.TASK
       state: NodeKind = NodeKind.START

       @cb.process
       def fire(self):
           if self.kind == NodeKind.TASK:
               ...
           self.state = NodeKind.EXCLUSIVE

   node = Node()
   node.kind = NodeKind.EXCLUSIVE                 # reads back as the member
   node.kind = cb.sweep(NodeKind.TASK, NodeKind.EXCLUSIVE)

* On the host, an assignment is checked against the enum when it is made.
  A member is stored as it is. An ``int`` that is one of the enum's values
  becomes that member (``node.kind = 3`` stores ``NodeKind.EXCLUSIVE``).
  Anything else, such as an invalid value, a float, or a member of another
  enum, raises ``ModelDefinitionError`` naming the field. Class defaults and
  the options of a sweep are checked the same way.
* In compiled code the field reads as a member, so it compares with members
  and converts with ``int()`` and ``float()``. A write must be a member of
  the field's enum. Compiled code can't turn an ``int`` into a member, so
  writing ``2`` is a compile error.
* An ``Output`` typed with an enum is collected like ``Output[int]``: its
  :class:`~cimba.Samples` hold the integer values as floats (``nan`` for a
  failed trial). Turn one back into a member with ``NodeKind(int(value))``.
  Without a default it starts at ``0``, even if no member has that value.
* ``cb.spawn(Node, kind=NodeKind.EXCLUSIVE)`` takes members for enum fields.
* A ``cb.decision`` can't be assigned to an enum ``Param``. Sweep the
  members instead.

Only ``enum.IntEnum`` is supported: a field typed with any other
``enum.Enum`` is a ``ModelDefinitionError`` that says to derive it from
``IntEnum``.

.. _array-fields:

Array fields
~~~~~~~~~~~~

A ``cb.Array[T]`` field gives each instance its own table of ``int``,
``float`` or ``bool`` values. Its element type is part of the class; its
length is configuration, so changing it never recompiles anything:

.. code-block:: python

   class Case(cb.Model):
       case_id: int = 0
       joins: cb.Array[int]                    # length given at spawn

   class Gate(cb.Model):
       counts: cb.Array[int] = cb.Array(4)     # four zeros
       weights: cb.Array[float] = cb.Array([0.5, 1.5, 2.0])
       join_slots: cb.Param[int] = 8

       @cb.process
       def run(self):
           case = cb.spawn(Case, case_id=1, joins=self.join_slots)
           case.joins[3] += 1
           total = 0
           for count in case.joins:
               total += count
           self.counts[0] = len(case.joins)

   gate = Gate()
   gate.counts = cb.Array(16)                         # a static model's length
   gate.counts = cb.sweep(cb.Array(8), cb.Array(16))  # or a sweep of lengths

* **Configuring.** ``cb.Array(n)`` is ``n`` zeros (``False`` for ``bool``);
  ``cb.Array([1, 2, 3])`` holds those initial values, checked against the
  element type. A static model's array is a class default or a host
  assignment, and may be a ``cb.sweep`` of arrays. A static model with an
  array field and neither is an error when the ``Experiment`` is created.
* **Spawning.** ``cb.spawn(Case, joins=n)`` takes an ``int`` *length* for an
  array field and gives a zeroed array of that length. Without the argument
  the class default is used (its length, and its values if it has any). A
  negative length fails the trial.
* **In compiled code** an array supports ``a[i]``, ``a[i] = v``,
  ``a[i] += 1``, ``len(a)`` and ``for x in a``, on ``self``, through a
  reference, a list item or a ``@cb.function`` parameter. Indices run from
  ``0`` to ``len(a) - 1``. Any other index, including a negative one, fails
  the trial with a reason such as ``Case.joins[8]: index out of range for
  length 8``; it never reads or writes outside the array. A write must fit
  the element type: a ``float`` into an ``int`` array is a compile error
  (convert with ``int()``). The field itself can't be reassigned in model
  code.
* **Every trial starts fresh.** A static array starts each trial from its
  configured contents, and each spawned model gets its own array. Arrays are
  per-trial state, not results: copy what you want to keep into an
  ``Output``.
* **Released models.** Releasing a spawned model frees its arrays at once.
  The model's record stays valid until the trial ends, so references to it
  still read its other fields, but its arrays then have length ``0`` and
  indexing them fails the trial. A static model keeps its arrays when
  released.

Plain field or output?
----------------------

Make a value an ``Output`` if it is a result, and a plain field if only the
logic needs it. A counter that is a result is an output that processes count
into directly:

.. code-block:: python

   class Checkout(cb.Model):
       served: cb.Output[int]                 # starts at 0
       revenue: cb.Output[float] = 0.0        # starts at 0.0
       mean_wait: cb.Output[float]            # nan unless something is recorded
       waits: cb.Dataset

       @cb.process
       def serve(self):
           ...
           self.served += 1
           self.revenue += price

       @cb.on_end
       def report(self):
           if self.waits.sample_count() > 0:
               self.mean_wait = self.waits.sample_mean()

Keep ``@cb.on_end`` for results computed at the end: entity statistics,
averages and ratios. A plain field copied into an output there is a sign the
field should have been the output.

Where values come from
----------------------

A field's value is resolved in this order:

1. an assignment on the instance (in ``__init__`` or later: ``model.x = 3``);
2. the default on the class (``x: cb.Param[float] = 1.0``).

Class defaults for entities, child models and sources are templates. Each
instance gets its own copy, so ``worker: cb.Resource = cb.Resource(capacity=2)``
gives every station its own two-unit resource. Entities with no value, such as
``queue: cb.Store[Job]``, are created automatically when the experiment is
defined. Create one yourself only when you need to configure it, for example
to ``capture()`` it.

Assignments are checked on the spot. Assigning a sweep to anything but a
``Param``, an input or a child model is a ``TypeError`` at that line, and so is assigning
something that isn't a source to an ``Input``. Missing values (a ``Param`` or
plain field with no value, an input with no source, a required ``Ref`` that is ``None``,
a reference to a model outside the tree) are reported when you create the
``Experiment``, with the instance's path in the message.

Composition: children, references and the tree
----------------------------------------------

Child fields and ``list[Model]`` fields form a **tree** rooted at the model
you pass to ``Experiment``. Every model in the tree is a **static instance**
and has a canonical **path** built from field names:

.. code-block:: text

   network                         the root (class name, lower-cased)
   network.facilities[0]           an item of a list field
   network.facilities[2].demand    a field of that item

Paths label things in error messages, ``describe()`` output, native process
names and input provenance. You never use them to *address* anything; you
use the objects.

References (``Ref``) are pointers into the same tree: to a parent, a sibling,
a cousin, anything. Cycles are fine. Two rules apply:

* a model instance may appear in the tree only **once** (to share it, own it
  in one place and reference it elsewhere);
* a reference must point to a model that **is** in the tree.

Build the tree in ``__init__``. It runs only in Python, when you construct
the model, and it's the natural place to turn domain arguments into fields
and children. Constructing a model never compiles or runs anything.

Override :meth:`cimba.Model.attached` to bind a child back to its owner. The hook
runs for each assigned child, including every option of a child sweep.

In compiled code, following a child or reference is a pointer load:
``self.plant.dispatcher.queue.put(job)``. A list supports ``len()`` and
indexing; loop over it with ``for i in range(len(self.items))``. Indices are
bounds-checked: an index outside the list, including a negative one, abandons
the trial with an error rather than reading stray memory. Following a
required reference that is missing does the same, and using an optional
reference that is ``None`` as a model fails the trial with ``a None
reference was used as a Station model``. Check it first:
``if self.next_station is not None:``.

Model code may **assign references** at run time. Children and lists are
structure and can't be reassigned, but a ``Ref`` field can point anywhere,
including at spawned models, which makes linked structures possible:

.. code-block:: python

   class Link(cb.Model):
       slot: int = 0
       next: cb.Ref["Link"] | None = None

   class Case(cb.Model):
       head: cb.Ref[Link] | None = None

       @cb.function
       def push(self, slot: int) -> None:
           link = cb.spawn(Link, slot=slot)
           head = self.head
           if head is not None:
               link.next = head
           self.head = link

* The value may be a model of the declared class or of a subclass (calls
  through the reference run the subclass's ``@cb.function`` implementations).
  Any other class is a compile error with the file and line of the
  assignment.
* ``None`` may be assigned to an optional reference only; assigning it to a
  required one is a compile error. An optional value (such as ``head``
  above) may be assigned to a required reference: a ``None`` there fails the
  trial, so check it first.
* A reference may point at a released spawned model. Its record stays valid
  until the trial ends, so the reference never dangles: its fields remain
  readable and writable, its processes never run again, and its arrays are
  gone (see :ref:`array-fields`). Use a field such as ``closed: bool`` if
  model code must tell released models apart.

Inheritance
-----------

A subclass inherits every field, process and hook, and may add fields or
override methods, including redeclaring a process with different
``copies``. A ``list[Base]`` (or ``Ref[Base]``) accepts instances of any
subclass. Through such a field, compiled code sees an item as ``Base``: it
can use every field ``Base`` declares, whatever the item's real class.

Each class is compiled once, no matter how many instances exist or how the
tree is wired. Changing structure, parameter values or input sources never
triggers recompilation.

Functions and polymorphism
--------------------------

Behavior that several processes share, or that should differ between
subclasses, goes in a ``@cb.function``:

.. code-block:: python

   class Router(cb.Model):
       @cb.function
       def lane(self, size: float) -> int:
           return 0

   class SizeRouter(Router):
       threshold: cb.Param[float] = 5.0

       @cb.function
       def lane(self, size: float) -> int:
           return 1 if size > self.threshold else 0

   class Dock(cb.Model):
       router: cb.Ref[Router]          # any subclass

       @cb.process
       def sort(self):
           ...
           lane = self.router.lane(parcel_size)

* Call it like a method, from processes, hooks, events, predicates or other
  functions: on ``self``, through a reference, on a list item, or on a model
  taken from a store.
* The **actual class of the model** decides which implementation runs, even
  when the caller sees only the base class. That's ordinary Python
  semantics, compiled to a per-class function table.
* Parameters and the return value are annotated with ``float``, ``int``,
  ``bool``, an ``enum.IntEnum`` or a model class (``None`` for no return
  value). A parameter typed with an enum takes members of that enum only; an
  ``int`` or ``float`` parameter also takes a member, as its value. An override must
  keep the signature and also be decorated.
* On the host it stays a normal method, so policy logic is easy to
  unit-test.

A child model can be **swept** over several objects,
``dock.router = cb.sweep(SizeRouter(), RoundRobin())``. Each option then gets
its own model tree, and each option's trials contain only that option. See
:doc:`../guides/policies`.

Static and dynamic instances
----------------------------

**Static** instances form the tree you build before the experiment. They
exist for the whole trial.

**Dynamic** instances are created during a trial with
``cb.spawn(ModelClass, field=value, ...)``, like customers, orders, parts and
ships, and removed with ``cb.release(model)``. They use the same model classes
with one restriction: a dynamic model holds values (plain fields,
``Param`` and ``Output``), arrays, references, inputs, processes, hooks and
functions, but no entities, children or lists. Give it a ``Ref`` to a static
model that owns the shared queues and resources, ``cb.Array`` fields for
per-instance tables sized at spawn, and ``Ref`` fields to link it to other
spawned models. See :doc:`processes` for the lifecycle.

Inspecting a configured model
-----------------------------

.. code-block:: python

   model.describe()
   # ({'label': 'mm1', 'class': 'MM1',
   #   'fields': (('interarrival', 'input'), ('queue', 'entity'), ...)},)

   cb.Experiment(model).describe()     # one row per field, with bound sources

   from cimba.diagrams import mermaid
   print(mermaid(model))               # the object graph as a Mermaid flowchart
