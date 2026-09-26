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

.. function:: mermaid(model)

   A Mermaid ``flowchart`` of the configured model tree: one node per static
   instance labelled ``path: Class``, solid edges for children, dotted edges
   for references.
