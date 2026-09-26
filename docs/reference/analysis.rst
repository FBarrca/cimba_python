Analysis, random and diagrams
=============================

cimba.analysis
--------------

.. module:: cimba.analysis

.. function:: summary(samples, *, confidence=0.95)

   Per design point statistics of a ``(points, replications)`` array (or
   :class:`~cimba.Samples`), ignoring non-finite values. Returns a
   :class:`Summary`.

.. class:: Summary

   Arrays with one entry per design point: ``n``, ``mean``, ``std``
   (sample, ``ddof=1``), ``lower`` and ``upper`` (Student-*t* confidence
   interval for the mean).

.. function:: compare(samples, *, a, b, confidence=0.95)

   Paired comparison of design points ``a`` and ``b``: the mean of
   ``samples[b] - samples[a]`` over replications where both are finite, with
   a Student-*t* interval. Returns a :class:`Comparison`.

.. class:: Comparison

   ``a``, ``b``, ``n`` (pairs used), ``difference``, ``lower``, ``upper``.
   ``nan`` if fewer than two pairs.

.. function:: check_input(source, *, reference, n=4096, seed=0, lags=10)

   Generate ``n`` values from ``source`` (a distribution, trace or row
   source) and compare them with the 1-D ``reference``. Returns an
   :class:`InputCheck`. For a ``bootstrap.joint`` source, ``reference`` is a
   mapping with the same keys or a 2-D array with one column per member, and
   the result is a :class:`JointInputCheck`.

.. class:: InputCheck

   ``reference_mean``, ``generated_mean``, ``reference_variance``,
   ``generated_variance``, ``reference_quantiles`` and ``generated_quantiles``
   (5, 25, 50, 75 and 95%), ``reference_acf``/``generated_acf`` and
   ``reference_pacf``/``generated_pacf`` (from lag 0), ``ks_statistic`` and
   ``ks_pvalue`` (two-sample Kolmogorov–Smirnov).

.. class:: JointInputCheck

   ``keys``, ``members`` (one :class:`InputCheck` per member),
   ``reference_correlation`` and ``generated_correlation`` matrices.

cimba.random
------------

.. module:: cimba.random

Distribution draws for **model-internal** randomness in compiled code (routing,
tie-breaking). They use the trial's default stream. For real-world
variability, prefer an input with an ``inputs.dist`` source. Calling these
on the host raises ``RuntimeError``.

.. list-table::
   :header-rows: 1

   * - Function
     - Returns
   * - ``uniform(min=0.0, max=1.0)``
     - float
   * - ``exponential(mean=1.0)``, ``erlang(k, mean)``, ``hypoexponential(means)``,
       ``hyperexponential(means, probabilities)``
     - float
   * - ``normal(mu=0.0, sigma=1.0)``, ``lognormal(m, s)``, ``logistic(m, s)``,
       ``cauchy(mode, scale)``, ``student_t(v, m=0.0, s=1.0)``
     - float
   * - ``gamma(shape, scale=1.0)``, ``beta(a, b, min=0.0, max=1.0)``,
       ``weibull(shape, scale)``, ``pareto(shape, mode)``, ``rayleigh(s)``,
       ``chi_squared(k)``, ``f_dist(a, b)``
     - float
   * - ``triangular(min, mode, max)``, ``pert(min, mode, max)``,
       ``pert_mod(min, mode, max, lambda_)``
     - float
   * - ``bernoulli(p)``
     - bool
   * - ``dice(min, max)``, ``geometric(p)``, ``binomial(n, p)``,
       ``negative_binomial(m, p)``, ``poisson(r)``, ``categorical(probabilities)``
     - int

``categorical(probabilities)`` returns an index into ``probabilities``.

cimba.diagrams
--------------

.. module:: cimba.diagrams

Draw configured models. Nothing is compiled or run, and input sources need
not be bound yet. See :doc:`../guides/diagrams`.

.. function:: structure(model)

   The object tree: one ``"instance"`` node per static model instance
   (labelled ``path: Class``), solid edges to children and list items, dotted
   edges for references. Returns a :class:`Graph`.

.. function:: process_graph(model, *, state=False, hooks=False)

   Interactions inferred from the source of every process and event,
   resolved against the configured objects. Nodes: processes, entities,
   inputs (with their source), events, and ``new X`` nodes for spawned model
   classes, grouped by model instance and by spawned class. Edges are
   labelled with the operation (``put``, ``get``, ``acquire``, ``release``,
   ``preempt``, ``enqueue``, ``cancel``, ``wait``, ``signal``, ``record``,
   ``next``, ``now``, ``at``, ``spawn``, ``schedule``, and ``write`` with
   ``state=True``) and point the way things flow. ``hooks=True`` adds
   ``on_start``/``on_end`` hooks as actors. Returns a :class:`Graph`.

.. function:: mermaid(model)

   Shortcut for ``structure(model).to_mermaid()``.

.. class:: Graph

   .. attribute:: nodes

      Tuple of :class:`Node`.

   .. attribute:: edges

      Tuple of :class:`Edge`.

   .. attribute:: groups

      Tuple of :class:`Group`.

   .. method:: node(key)

      The node with that key.

   .. method:: to_mermaid(direction="TD")

      Mermaid ``flowchart`` text (``TD``, ``LR``, …); groups become
      subgraphs.

   .. method:: to_dot(rankdir="TB")

      Graphviz DOT text; groups become clusters (dashed for spawned classes).

   .. method:: topological_order()

      Node keys in flow order; ``ValueError`` if the graph has a cycle.

.. class:: Node

   ``key`` (a canonical path such as ``harbor.facilities.tugs`` or
   ``Ship.voyage``), ``label``, ``kind`` (``instance``, ``process``,
   ``hook``, ``event``, ``entity``, ``input``, ``state`` or ``model``) and
   ``group``.

.. class:: Edge

   ``source``, ``target``, ``label`` and ``style`` (``"solid"`` or
   ``"dotted"``).

.. class:: Group

   ``key``, ``label`` and ``kind`` (``"model"`` or ``"spawned"``).
