Input sources: cimba.inputs
===========================

.. module:: cimba.inputs

``from cimba import inputs``. Every function here returns an immutable
**source** to bind to an ``Input`` or ``Series`` field. Sources are
configuration only; they never appear in model code. All parameters are
keyword-only unless shown otherwise.

Distributions: ``inputs.dist``
------------------------------

Drawn inside the trial on the input's own random stream. Never exhausted.
Numeric parameters may be :func:`cimba.sweep` objects (or ``sweep.map(...)``);
fixed values are validated when the source is built, swept values when the
experiment is created.
Every constructor accepts keyword-only ``tag: str | None = None``. A tag
replaces the input's tree path when deriving its random stream, preserving
draws across renames and moves. Equal tags intentionally share a stream
identity; the default retains path-based seeding.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Function
     - Distribution
   * - ``exponential(mean)``
     - Exponential with the given mean (not rate).
   * - ``erlang(k, mean)``
     - Sum of ``k`` exponentials; overall ``mean``.
   * - ``hypoexponential(means)``
     - Sum of exponentials with the given means (positional list).
   * - ``hyperexponential(means, probabilities)``
     - Mixture of exponentials (positional lists).
   * - ``gamma(shape, scale=1.0)``
     - Gamma; mean ``shape * scale``.
   * - ``weibull(shape, scale=1.0)``
     - Weibull.
   * - ``lognormal(mean=0.0, sigma=1.0)``
     - ``exp(N(mean, sigma))``: parameters of the underlying normal.
   * - ``normal(mean=0.0, sd=1.0)``
     - Normal.
   * - ``logistic(mean=0.0, scale=1.0)``
     - Logistic.
   * - ``cauchy(mode=0.0, scale=1.0)``
     - Cauchy.
   * - ``student_t(df, mean=0.0, scale=1.0)``
     - Location-scale Student *t*.
   * - ``chi_squared(df)``
     - Chi-squared.
   * - ``f_dist(df1, df2)``
     - F.
   * - ``rayleigh(scale)``
     - Rayleigh.
   * - ``pareto(shape, mode=1.0)``
     - Pareto with minimum ``mode``.
   * - ``uniform(low=0.0, high=1.0)``
     - Continuous uniform.
   * - ``triangular(low, mode, high)``
     - Triangular.
   * - ``pert(low, mode, high)``
     - PERT (scaled beta, weight 4).
   * - ``pert_mod(low, mode, high, weight)``
     - Modified PERT with custom weight.
   * - ``beta(a, b, low=0.0, high=1.0)``
     - Beta scaled to ``[low, high]``.
   * - ``bernoulli(p)``
     - 1 with probability ``p``, else 0.
   * - ``binomial(n, p)``
     - Binomial.
   * - ``geometric(p)``
     - Trials until first success (≥ 1).
   * - ``negative_binomial(successes, p)``
     - Failures before ``successes`` successes.
   * - ``poisson(mean)``
     - Poisson.
   * - ``dice(low, high)``
     - Integer uniform on ``low..high`` inclusive.
   * - ``categorical(values, probabilities)``
     - ``values[i]`` with probability ``probabilities[i]`` (positional lists,
       summing to 1).

Integer-valued distributions work well with ``cb.Input[int]``.

Recorded data
-------------

.. function:: trace(data, *, on_exhausted="fail")

   Replay a 1-D array exactly as recorded. ``data`` is copied and
   fingerprinted (SHA-256). ``on_exhausted`` is ``"fail"``, ``"wrap"`` or
   ``"end_trial"``.

Bootstrap resampling: ``inputs.bootstrap``
------------------------------------------

Generated on the host once per replication from a stream derived from the
replication seed and the input's identity. All are prefix-stable and extend
transparently.

.. function:: bootstrap.iid(data)

   Independent draws with replacement. Keeps the marginal distribution and
   drops autocorrelation.

.. function:: bootstrap.moving_block(data, *, block)

   Concatenate random blocks of ``block`` consecutive values.

.. function:: bootstrap.circular_block(data, *, block)

   Like ``moving_block``, but blocks may wrap around the end of the data.

.. function:: bootstrap.stationary(data, *, mean_block)

   Stationary bootstrap: blocks of geometrically distributed length with mean
   ``mean_block``, wrapping around. The resampled series is stationary.

.. function:: bootstrap.joint(panel, *, mean_block, tag=None)

   Stationary bootstrap of several equal-length series that uses the **same**
   blocks for every member, preserving cross-correlation. ``panel`` is a
   mapping or a sequence of 1-D arrays. Index the result like the panel,
   ``joint[key]``, to get each member's source. Iterating yields the keys.

.. function:: bootstrap.intermittent(data, *, jitter=False)

   For intermittent series (mostly zeros): a two-state Markov chain for
   occurrence, fitted to the data, with sizes resampled from the nonzero
   values. ``jitter=True`` perturbs sizes. Needs at least two nonzero values
   and one zero.

Fitted time-series models: ``inputs.fitted``
--------------------------------------------

These separate deterministic **structure** (trend and season) from the
residuals, and generate new series as structure plus resampled or modelled
residuals. Common arguments:

``trend``
   Polynomial degree of the trend (``0`` constant, ``1`` linear, …), ``None``
   for the mean, or ``"auto"`` to choose among 0, 1 and 2 by AICc.
``period``
   Season length in samples, ``None`` for no season, or ``"auto"`` to detect
   one from the periodogram. With a period, structure comes from an STL
   decomposition (``robust=True`` for robust STL); beyond the data, the season
   repeats and the trend extrapolates linearly.
``start``
   Index in the reference series where generated rows begin.
``nonnegative``
   Clip generated values at zero (demand, counts).

.. function:: fitted.residual(data, *, trend=1, period=None, mean_block=None, start=0, nonnegative=False, robust=False)

   Structure plus residuals resampled iid, or with a stationary bootstrap if
   ``mean_block`` is given.

.. function:: fitted.wild(data, *, trend=1, period=None, weights="rademacher", start=0, nonnegative=False, robust=False)

   Wild bootstrap: structure plus each in-sample residual multiplied by a
   random weight (``"rademacher"``, ``"mammen"`` or ``"normal"``). Keeps
   heteroscedasticity. Limited to ``len(data) - start`` values, and fails
   rather than extends beyond that.

.. function:: fitted.sieve(data, *, order=None, trend=None, period=None, start=0, nonnegative=False, robust=False)

   Sieve bootstrap: fit an AR(``order``) model to the residuals (order
   chosen by AIC when ``None``) and regenerate them from resampled
   innovations. Needs at least eight observations.

.. function:: fitted.intermittent(data, *, jitter=False)

   Same as :func:`bootstrap.intermittent`.

Fitting distributions
---------------------

.. function:: fit(data, candidates=None)

   Fit candidate distributions by maximum likelihood and return a tuple of
   :class:`FitResult`, best (lowest AIC) first. The default candidates are
   ``dist.exponential``, ``dist.gamma``, ``dist.lognormal``, ``dist.normal``
   and ``dist.weibull``. Candidates the data can't support (for example
   non-positive values for a gamma) are skipped.

.. class:: FitResult

   .. attribute:: source

      The fitted :class:`DistributionSource`, ready to bind.

   .. attribute:: aic
   .. attribute:: ks

      Kolmogorov–Smirnov distance between the data and the fit.

   .. attribute:: log_likelihood

Custom sources
--------------

.. function:: row_source(method, data, draw, *, parameters=None, tag=None, prefix_stable=True, max_length=None, length_hint=None, on_exhausted="extend")

   Build a row source from ``draw(rng, length) -> array``. See
   :doc:`../guides/custom_sources` for the contract.

.. function:: trace_rng(seed, tag)

   The ``numpy.random.Generator`` Cimba would pass to a row source with
   ``tag`` for a trial seed. Useful for testing prefix stability.

Source objects
--------------

All sources have ``describe() -> dict`` (versioned provenance) and an
``on_exhausted`` attribute.

.. class:: DistributionSource

   ``method``, ``parameters`` and optional ``tag``. Built by ``inputs.dist.*``.

.. class:: TraceSource

   ``values`` (read-only), ``fingerprint`` and ``on_exhausted``. Built by
   :func:`trace`.

.. class:: GeneratedRows

   A row source. ``generate(rngs, length)`` returns a
   ``(len(rngs), length)`` array; also has ``prefix_stable``,
   ``max_length``, ``length_hint`` and ``tag``. Built by the bootstrap,
   fitted and :func:`row_source` functions.

.. class:: RowSource

   The protocol ``GeneratedRows`` implements.

.. class:: Source

   The protocol every source implements.

.. exception:: InputError

   Raised for invalid source parameters or data, and for generators that
   return malformed rows.
