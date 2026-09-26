Random Draws
============

Cimba gives every trial its own pseudo-random stream. Pass ``seed=...`` to
``model.experiment(...)`` when a run should be reproducible; leave it unset
when Cimba should choose independent seeds for you. In model code, import
``cimba.random`` alongside ``cimba.sim`` and draw from that per-trial stream:

.. code-block:: python

   import cimba.random as random
   import cimba.sim as sim

   class Clinic(sim.Model):
       mean_interarrival: sim.Param
       mean_service: sim.Param
       served: sim.Output
       queue: sim.Queue

       @sim.process
       def arrivals(self: "Clinic"):
           while True:
               sim.hold(random.exponential(self.mean_interarrival))
               self.queue.put(1)

       @sim.process
       def server(self: "Clinic"):
           while True:
               self.queue.get(1)
               sim.hold(random.gamma(shape=2.0, scale=self.mean_service / 2.0))

   model = Clinic()

``cimba.random`` draws are available inside compiled model callbacks and the
``@numba.njit`` helpers those callbacks invoke. Calling them from ordinary
Python raises ``RuntimeError``. For preprocessing and other Python-side draws,
use ``numpy.random.default_rng(seed)``; trace generators receive their own NumPy
generator from the experiment.

The random API is intentionally namespaced. Flat spellings such as
``sim.exponential(...)``, ``sim.random.exponential(...)``, or
``cimba.exponential(...)`` are not part of the public API.

Parameter Conventions
---------------------

The distribution functions use native parameter names rather than abbreviated
legacy aliases. In particular:

* ``exponential(mean=1.0)`` uses a mean, not a rate.
* ``normal(mu=0.0, sigma=1.0)`` uses mean and standard deviation.
* ``gamma(shape, scale=1.0)`` uses shape and scale.
* ``beta(a, b, min=0.0, max=1.0)`` returns a beta variate scaled to
  ``[min, max]``.
* ``student_t(v, m=0.0, s=1.0)`` uses degrees of freedom ``v``, location
  ``m``, and scale ``s``.

Keyword arguments are supported in compiled model callbacks and their
``@numba.njit`` helpers:

.. code-block:: python

   class Clinic(sim.Model):
       @sim.process
       def customer(self: "Clinic"):
           patience = random.triangular(min=0.5, mode=1.0, max=2.0)
           priority = 5 if random.bernoulli(p=0.25) else 0
           sim.hold(random.normal(mu=patience, sigma=0.1))

Continuous Draws
----------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Function
     - Meaning
   * - ``uniform(min=0.0, max=1.0)``
     - Continuous uniform draw between ``min`` and ``max``.
   * - ``exponential(mean=1.0)``
     - Exponential draw with the given mean interarrival or service time.
   * - ``gamma(shape, scale=1.0)``
     - Gamma draw with positive ``shape`` and ``scale``.
   * - ``normal(mu=0.0, sigma=1.0)``
     - Normal draw with mean ``mu`` and standard deviation ``sigma``.
   * - ``lognormal(m, s)``
     - Lognormal draw where the underlying normal has location ``m`` and
       positive scale ``s``.
   * - ``logistic(m, s)``
     - Logistic draw with location ``m`` and positive scale ``s``.
   * - ``cauchy(mode, scale)``
     - Cauchy draw with the given mode and positive scale.
   * - ``rayleigh(s)``
     - Rayleigh draw with positive scale ``s``.
   * - ``weibull(shape, scale)``
     - Weibull draw with positive shape and scale.
   * - ``pareto(shape, mode)``
     - Pareto draw with positive shape and positive mode/minimum.
   * - ``beta(a, b, min=0.0, max=1.0)``
     - Beta draw scaled from ``[0, 1]`` into ``[min, max]``.
   * - ``erlang(k, mean)``
     - Erlang draw with integer shape ``k`` and positive mean.
   * - ``hypoexponential(means)``
     - Sum of independent exponential stages, one for each positive mean in
       ``means``.
   * - ``hyperexponential(means, probabilities)``
     - Choose one positive mean according to ``probabilities``, then draw one
       exponential with that mean.
   * - ``pert(min, mode, max)``
     - PERT draw for bounded expert estimates with minimum, most-likely value,
       and maximum.
   * - ``pert_mod(min, mode, max, lambda_)``
     - Modified PERT draw; larger ``lambda_`` concentrates more mass near
       ``mode``.
   * - ``chi_squared(k)``
     - Chi-squared draw with positive degrees of freedom ``k``.
   * - ``f_dist(a, b)``
     - F distribution draw with positive numerator and denominator degrees of
       freedom.
   * - ``student_t(v, m=0.0, s=1.0)``
     - Student's t draw with positive degrees of freedom ``v``, location
       ``m``, and positive scale ``s``.

Discrete Draws
--------------

Discrete functions produce integer-valued draws. Use ``bernoulli(p)`` directly
in a condition to choose an outcome with probability ``p``.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Function
     - Meaning
   * - ``bernoulli(p)``
     - Success with probability ``p``, suitable for an ``if`` condition.
   * - ``dice(min, max)``
     - Integer draw from the inclusive range ``[min, max]``.
   * - ``poisson(r)``
     - Poisson draw with positive rate/mean ``r``.
   * - ``geometric(p)``
     - Number of trials until the first success, using success probability
       ``p``.
   * - ``binomial(n, p)``
     - Number of successes in ``n`` independent Bernoulli trials.
   * - ``negative_binomial(m, p)``
     - Number of failures before ``m`` successes.
   * - ``categorical(probabilities)``
     - Zero-based index sampled from non-negative probabilities that sum to
       ``1.0``.

Probability Vectors
-------------------

``categorical()`` and ``hyperexponential()`` accept Python sequences, tuples, or
NumPy arrays. Probabilities must be finite and non-negative and must sum to
``1.0``. Zero entries are allowed. The selected index is zero-based, which makes
it convenient for arrays:

.. code-block:: python

   DESTINATION = (0.55, 0.30, 0.15)
   WALK_TIME = (2.0, 5.0, 12.0)

   class Park(sim.Model):
       @sim.process
       def visitor(self: "Park"):
           i = random.categorical(DESTINATION)
           sim.hold(WALK_TIME[i])

Seeds And Reproducibility
-------------------------

Control simulation streams with the experiment-level seed:

.. code-block:: python

   exp = model.experiment(replications=20, duration=1_000.0, seed=20260705)
   exp.run()

That seed is expanded into independent per-trial streams, so parallel execution
stays reproducible. Re-running an experiment resets each trial to its original
seed. There is no public thread-local reseeding API.

Use ``sim.Param`` fields when distribution parameters should vary across design
points. Use NumPy generators to create exogenous inputs before the run, and pass
those inputs through ``sim.Trace``.
