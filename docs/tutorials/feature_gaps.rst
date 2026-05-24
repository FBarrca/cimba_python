Feature Gaps
============

The completed C tutorials can mostly be expressed with the Python API. The
migration found these native C tutorial features that are not currently exposed
as Python bindings:

* Native event pattern find/count/cancel helpers. Python exposes direct event
  scheduling, cancellation, rescheduling, event waiting, and reprioritization,
  but not wildcard pattern matching over action/subject/object triples.
* User-level native logger calls such as ``cmb_logger_user()``. Python exposes
  logger flag controls, but not formatted user log records with native
  timestamp/process/function prefixes.
* Native ``FILE *`` text report printers are intentionally not exposed. Use the
  structured :mod:`cimba.reporting` helpers instead:

  .. code-block:: python

     report = cimba.reporting.history_report(history, lags=20)
     print(cimba.reporting.format_report(report))

  Install ``cimba[plot]`` to use helpers such as
  :func:`cimba.reporting.plot_report`.
* Internal C containers such as ``cmi_hashheap`` and ``cmi_slist``. Python code
  should normally use dictionaries, lists, sets, :class:`cimba.ObjectQueue`, or
  :class:`cimba.PriorityQueue`.
* High-level CUDA/thread-stream integration for the work-in-progress GPU
  tutorial. The low-level native thread hook capsule API exists, but there is no
  Python tutorial API for CUDA device assignment.
