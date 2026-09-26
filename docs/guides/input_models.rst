Choose and check an input model
===============================

**Goal:** go from raw observations to a source you can defend.

Look at the data first
----------------------

A few questions decide the family of source:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - If the data…
     - consider
   * - are continuous, independent and you want a closed form
     - ``inputs.fit(data)`` → a ``dist.*`` source
   * - are small integers (counts, extra days)
     - ``inputs.bootstrap.iid(data)``, or ``dist.poisson`` /
       ``dist.categorical``
   * - are autocorrelated (today looks like yesterday)
     - ``bootstrap.stationary``, ``moving_block`` or ``circular_block``
   * - have trend or seasonality you want to keep or extrapolate
     - ``fitted.residual`` (structure + resampled residuals),
       ``fitted.sieve`` (structure + AR model)
   * - are mostly zeros with occasional values (intermittent demand)
     - ``bootstrap.intermittent``
   * - are several related series (sites, products)
     - ``bootstrap.joint`` to keep their co-movement

Fit distributions
-----------------

:func:`cimba.inputs.fit` fits candidates by maximum likelihood and returns
them ranked by AIC, each with its Kolmogorov–Smirnov distance:

.. code-block:: python

   from cimba import inputs

   for candidate in inputs.fit(service_times):
       print(candidate.source.describe()["method"],
             round(candidate.aic, 1), round(candidate.ks, 3))

.. code-block:: text

   dist.exponential 974.7 0.027
   dist.weibull     976.6 0.023
   dist.gamma       976.7 0.026
   dist.lognormal  1035.8 0.071
   dist.normal     1419.2 0.165

The default candidates are exponential, gamma, lognormal, normal and
Weibull; pass ``candidates=[inputs.dist.gamma, inputs.dist.weibull]`` to
restrict them. Candidates that don't support the data (a gamma with zeros)
are skipped. Each result's ``.source`` is ready to bind:

.. code-block:: python

   best = inputs.fit(service_times)[0]
   model.service_time = best.source

Here exponential, Weibull and gamma are within two AIC points of each other:
the data can't tell them apart, and the simplest, the exponential, is a
fine choice.

Check before you trust
----------------------

:func:`cimba.analysis.check_input` draws a sample from any source (``n``
values, with its own ``seed``) and compares it with reference data:

.. code-block:: python

   from cimba import analysis

   source = inputs.bootstrap.stationary(history, mean_block=7)
   report = analysis.check_input(source, reference=history, n=4096, lags=10)

   report.reference_mean, report.generated_mean
   report.reference_variance, report.generated_variance
   report.reference_quantiles, report.generated_quantiles   # 5, 25, 50, 75, 95 %
   report.reference_acf, report.generated_acf               # lags 0..10
   report.reference_pacf, report.generated_pacf
   report.ks_statistic, report.ks_pvalue

What to look for:

* **Means and quantiles** agree: the marginal distribution is right.
* **ACF/PACF** agree: the time structure is right. An iid bootstrap of
  autocorrelated data shows near-zero ACF where the reference has
  structure. That tells you to use a block method.
* A **small KS statistic** says the distributions are close. With thousands of
  samples even tiny differences become "significant", so read the statistic,
  not just the p-value.

For a joint source, pass the joint object and a panel (a dict keyed like
the joint, or a 2-D array). The report adds cross-correlation matrices and a
per-member report:

.. code-block:: python

   joint = inputs.bootstrap.joint(panel, mean_block=14)
   report = analysis.check_input(joint, reference=panel)
   report.reference_correlation, report.generated_correlation
   report.members[0].ks_statistic

Choosing block lengths
----------------------

Block bootstraps keep the dependence *within* a block and break it between
blocks. Longer blocks keep more structure but give less variety. A common
starting point is a mean block of about *n*\ :sup:`1/3` for *n* observations
(22 for 10,000 days). Then confirm with ``check_input`` that the ACF matches
at the lags that matter to your model.

Let the study decide
--------------------

When two input models are both plausible, run both and see whether the
answer changes. See :doc:`comparing`.
