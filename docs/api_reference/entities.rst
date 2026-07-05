Queues, Resources and Conditions
================================

Many simulation entities share one pattern: a process makes a demand, the
demand is satisfied immediately or the process waits, and a later state change
wakes one or more waiting processes.

Every declared entity field carries its verbs as methods, so a queue field
``env.queue`` is used as ``env.queue.put(1)``, ``env.queue.get(1)``, and so
on; a component-owned field is used the same way through ``self.<field>``
inside the component's own methods.

* ``sim.Queue`` stores counts and supports ``.put()`` and ``.get()``.
* ``sim.Resource`` represents one exclusive unit with ``.acquire()`` and
  ``.release()``.
* ``sim.Pool`` represents multiple interchangeable units with
  ``.acquire()`` and ``.release()``, both taking the requested amount.
* ``sim.Store`` and ``sim.PQueues`` hold integer payloads taken by handle or
  priority.
* ``sim.Condition`` combines ``sim.Predicate`` functions, ``.wait_for()``,
  and ``.signal()`` for arbitrary model-defined readiness checks.

Queues and resources
--------------------

``env.<queue>.put(amount)``, ``.get(amount)``, ``.level()``, ``.space()``,
``.mean_level()``.

``env.<resource>.acquire()``, ``.release()``, ``.preempt()``,
``.available()``, ``.in_use()``, ``.held(process)``, ``.mean_in_use()``.

``env.<pool>.acquire(amount)``, ``.release(amount)``, ``.preempt(amount)``,
``.available()``, ``.in_use()``, ``.held(process)``, ``.mean_in_use()``.

Every queue/resource/pool also has ``.report()`` and
``.report_file(path, append=1)`` for the native text report. Each entity's
recorded time-weighted history is ``env.<entity>.history()``; see
:doc:`data` for the chained ``.history().mean()``-style summary calls.

Stores, priority queues and conditions
--------------------------------------

``env.<store>.put(obj)``, ``.get()`` (returns ``(status, obj)``),
``.take()``, ``.length()``, ``.space()``, ``.position(obj)``,
``.mean_length()``, ``.report()``, ``.report_file(path, append=1)``.

A ``sim.PQueues`` field is indexed to reach one priority queue:
``env.<pqueues>[i].put(obj, priority)``, ``.get()``, ``.take()``,
``.length()``, ``.space()``, ``.position(entry)``,
``.reprioritize(entry, priority)``, ``.cancel(entry)``, ``.mean_length()``,
``.report()``, ``.report_file(path, append=1)``.

``env.<condition>.signal()`` wakes the condition's waiters;
``env.<condition>.wait_for(predicate)`` blocks until ``predicate`` (an
``env.<name>`` field bound by ``@model.predicate``) is satisfied, and is
re-evaluated on every ``.signal()``.

Stores and priority-queue elements have ``.history()`` too:
``env.<store>.history()`` and ``env.<pqueues>[i].history()``.
