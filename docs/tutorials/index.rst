Tutorial: Modeling with Cimba Python
====================================

These pages are the Python-API migration of the upstream Cimba tutorial. They
follow the same tutorial order and modeling story as the C docs, but replace
manual C object lifecycles, pointers, structs, and native callbacks with the
public Python API.

The runnable sources live in ``tutorial/`` and are covered by
``tests/tutorial/``. The text here keeps close to the C tutorial's structure and
calls out places where Python deliberately differs or where a native feature is
not yet bound.

.. toctree::
   :maxdepth: 2

   mm1_queue
   resources_interruptions
   agents_queues
   harbor_conditions
   gpu_physics
   feature_gaps
