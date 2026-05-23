The Cimba C Library
===================

The native C library is the engine behind Cimba Python. It owns the event queue,
process scheduler, coroutine context switching, native resources, and random
number generators.

Use the C docs when you need the deeper background:

* `Cimba C documentation`_
* `Cimba C tutorial`_
* `Cimba C background`_
* `Cimba C API reference`_
* `Cimba C installation guide`_

The vendored source for those docs is in ``subprojects/cimba/docs``. These
Python docs follow the same Sphinx and Read the Docs theme setup, but are
organized for Python users.

Mapping Python to Cimba C
-------------------------

.. list-table::
   :header-rows: 1

   * - Python API
     - Native concept
   * - :class:`cimba.Simulation`
     - Event queue, simulation clock, and random generator
   * - :class:`cimba.Process`, :func:`cimba.hold`
     - Stackful Cimba processes and process scheduling
   * - :class:`cimba.Buffer`
     - Numeric buffers with blocking put/get
   * - :class:`cimba.ObjectQueue`, :class:`cimba.PriorityQueue`
     - Object queues and priority queues
   * - :class:`cimba.Resource`, :class:`cimba.ResourcePool`
     - Binary and counting resources
   * - Random functions
     - Cimba random number generators and distributions
