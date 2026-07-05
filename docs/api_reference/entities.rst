Queues, Resources and Conditions
================================

Many simulation entities share one pattern: a process makes a demand, the
demand is satisfied immediately or the process waits, and a later state change
wakes one or more waiting processes.

* ``sim.Queue`` stores counts and supports ``sim.put()`` and ``sim.get()``.
* ``sim.Resource`` represents one exclusive unit with ``sim.acquire()`` and
  ``sim.release()``.
* ``sim.Pool`` represents multiple interchangeable units with
  ``sim.pool_acquire()`` and ``sim.pool_release()``.
* ``sim.Store`` and ``sim.PQueues`` hold integer payloads taken by handle or
  priority.
* ``sim.Condition`` combines ``sim.Predicate`` functions, ``sim.wait_for()``,
  and ``sim.signal()`` for arbitrary model-defined readiness checks.

Queues and resources
--------------------

``put()``, ``get()``, ``level()``, ``space()``, ``mean_level()``,
``acquire()``, ``release()``, ``preempt()``, ``available()``, ``in_use()``,
``held()``, ``mean_in_use()``, ``pool_acquire()``, ``pool_release()``,
``pool_preempt()``, ``pool_available()``, ``pool_held()``, ``pool_in_use()``,
``pool_mean_in_use()``, ``queue_report()``, ``queue_report_file()``,
``resource_report()``, ``resource_report_file()``, ``pool_report()``,
``pool_report_file()``. Each entity's recorded time-weighted history is
``env.<entity>.history()``; see :doc:`data` for the chained
``.history().mean()``-style summary calls.

Stores, priority queues and conditions
--------------------------------------

``store_put()``, ``store_get()``, ``store_take()``, ``store_length()``,
``store_space()``, ``store_position()``, ``store_mean_length()``,
``store_report()``, ``store_report_file()``,
``pq_put()``, ``pq_get()``, ``pq_take()``, ``pq_length()``, ``pq_space()``,
``pq_position()``, ``pq_reprioritize()``, ``pq_cancel()``,
``pq_mean_length()``, ``pq_report()``, ``pq_report_file()``,
``wait_for()``, ``signal()``. Stores and priority-queue elements have
``.history()`` too: ``env.<store>.history()`` and
``env.<pqueues>[i].history()``.
