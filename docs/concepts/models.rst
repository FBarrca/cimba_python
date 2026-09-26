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
       busy_since: cb.State[float] = 0.0                                   # state
       served: cb.Output[int]                                              # output
       lane: int = 0                                                       # constant
       queue: cb.Store[Job]                                                # entity
       worker: cb.Resource = cb.Resource(capacity=2)                       # configured entity
       plant: cb.Ref["Plant"]                                              # reference
       next_station: cb.Ref["Station"] | None = None                       # optional reference

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
   * - ``x: cb.State[T] = v``
     - state
     - A value that processes change during a trial. It starts at ``v`` in
       every trial, so an initial value is required for static models.
   * - ``x: cb.Output[T]``
     - output
     - A per-trial result, usually written in an ``@cb.on_end`` hook. It
       starts as ``nan`` (``float``) or ``0`` (``int``, ``bool``).
       Collected into :class:`~cimba.Samples`.
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
   * - ``x: int = 3``
     - constant
     - A plain annotated scalar: fixed for the model instance, never swept.
   * - (no annotation)
     - host-only
     - An ordinary Python attribute, such as a display name. Compiled code
       never sees it.

Scalar field types (``T``) are ``float``, ``int`` and ``bool``. Strings,
lists of numbers and other Python objects can't live in trial memory. Keep
them as host-only attributes, or as module-level constants (NumPy arrays work
well) that compiled code reads.

Where values come from
----------------------

A field's value is resolved in this order:

1. an assignment on the instance (in ``__init__`` or later: ``model.x = 3``);
2. the default on the class (``x: cb.Param[float] = 1.0``).

Class defaults for entities and sources are templates. Each instance gets its
own copy, so ``worker: cb.Resource = cb.Resource(capacity=2)`` gives every
station its own two-unit resource. Entities with no value, such as
``queue: cb.Store[Job]``, are created automatically when the experiment is
defined. Create one yourself only when you need to configure it, for example
to ``capture()`` it.

Assignments are checked on the spot. Assigning a sweep to anything but a
``Param``, an input or a child model is a ``TypeError`` at that line, and so is assigning
something that isn't a source to an ``Input``. Missing values (a ``Param``
with no value, an input with no source, a required ``Ref`` that is ``None``,
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

In compiled code, following a child or reference is a pointer load:
``self.plant.dispatcher.queue.put(job)``. A list supports ``len()`` and
indexing; loop over it with ``for i in range(len(self.items))``. Indices are
bounds-checked: an index outside the list, including a negative one, abandons
the trial with an error rather than reading stray memory. Following a
required reference that is missing does the same.

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
  ``bool`` or a model class (``None`` for no return value). An override must
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
with one restriction: a dynamic model holds values (``Param``, ``State``,
``Output``, constants), references, inputs, processes, hooks and functions,
but no entities or children. Give it a ``Ref`` to a static model that owns the
shared queues and resources. See :doc:`processes` for the lifecycle.

Inspecting a configured model
-----------------------------

.. code-block:: python

   model.describe()
   # ({'label': 'mm1', 'class': 'MM1',
   #   'fields': (('interarrival', 'input'), ('queue', 'entity'), ...)},)

   cb.Experiment(model).describe()     # one row per field, with bound sources

   from cimba.diagrams import mermaid
   print(mermaid(model))               # the object graph as a Mermaid flowchart
