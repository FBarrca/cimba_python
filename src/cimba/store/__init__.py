"""AST-lowering support behind the object-oriented entity method sugar.

``sim.Queue``/``sim.Resource``/``sim.Pool``/``sim.Store``/``sim.PQueues``/
``sim.Condition`` fields are opaque int64 handles in the trial record, but
process bodies write to them with plain method syntax, e.g.
``env.queue.put(1)``. This package is the machinery behind that sugar,
mirroring ``cimba._dataset`` and ``cimba._timeseries``:

* ``helpers`` -- thin njit-callable wrappers over the native bindings, one
  per supported method, grouped by entity kind;
* ``methods`` -- the per-kind method tables and the AST lowerers that
  rewrite ``env.<entity>.method(...)`` (and, for Condition, the implicit
  ``env`` argument to ``wait_for``) into calls to those helpers, used by
  both ``_model.py`` (plain model fields) and ``_components.py``
  (component-owned fields, including component-collection paths).

Nothing here is part of the public API; see ``cimba.sim`` for the verbs
and entity types model authors actually write.
"""
