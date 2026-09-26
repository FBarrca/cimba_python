Quickstart
==========

This complete queue model consumes one interarrival and one service time per
job. The ``Container`` counts waiting jobs and records its mean level during
the measurement window.

.. literalinclude:: ../tutorial/tut_1_1.py
   :language: python

Run it from the repository root with ``uv run python tutorial/tut_1_1.py``.
The following tutorials build toward sweeps, preemption, dynamic models,
harbor operations, manufacturing, and inventory. Every script in
``tutorial/`` stands alone.
