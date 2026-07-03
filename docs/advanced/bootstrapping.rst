Bootstrapping Methods
=====================

Data-driven simulation replaces fitted input distributions with resamples of
observed data. This page surveys the bootstrap methods available for
generating those resamples and how to choose among them. The mechanics of
feeding a resample into a trial -- trace fields, callable generators, seeds --
are covered in :doc:`traces`; this page is about which resampler to use.

The methods are organized by how much temporal structure the data has. That
is the axis that matters: preserving autocorrelation is the whole reason
traces are generated outside the simulation and replayed, rather than
resampled draw-by-draw inside it.

Independent data: the ordinary bootstrap
----------------------------------------

The ordinary (Efron) bootstrap resamples individual observations with
replacement. It is correct when observations really are independent --
service times, order sizes, repair durations:

.. code-block:: python

   from cimba import bootstrap

   service = bootstrap.iid(observed_service_times, length=500)

It is wrong for anything with serial dependence, because independent draws
shuffle away the autocorrelation. A demand history with weekly structure,
resampled point by point, produces trajectories with the right marginal
distribution and none of the temporal behavior the model is supposed to
stress.

Stationary dependent series: block bootstraps
---------------------------------------------

The workhorse family for demand-like data. Instead of resampling points,
block bootstraps resample contiguous blocks of the series, so dependence
within each block survives:

* **Moving block bootstrap** (``bootstrap.moving_block``) -- draw fixed-length
  blocks uniformly from all overlapping windows and concatenate them. Simple
  and well-studied.
* **Circular block bootstrap** (``bootstrap.circular_block``) -- the same, but
  blocks wrap around the end of the series, so observations near the edges
  are not underweighted.
* **Stationary bootstrap** (``bootstrap.stationary``) -- block lengths are
  random (geometric with a chosen mean), which makes the resampled series
  itself stationary instead of having artificial seams every fixed number of
  steps. Often the best default.
* **Tapered and matched variants** -- down-weight block edges to reduce seam
  bias. More refinement than most models need to start with; not provided by
  ``cimba.bootstrap``.

The key tuning knob is block length: too short destroys the dependence being
preserved, too long reduces diversity across resamples. A common rule of
thumb is ``n ** (1/3)`` for a series of ``n`` observations; automatic
selection (Politis--White) is the principled alternative when it matters.

All block methods assume stationarity. If the series trends or has seasonal
structure, resampling blocks of it mixes regimes; decompose first and
bootstrap what is left, as described next.

Trend and seasonality: residual and model-based bootstraps
----------------------------------------------------------

For non-stationary data, fit the structure and bootstrap the residuals: model
the trend and seasonality, resample the residuals -- i.i.d. or block,
depending on what dependence remains in them -- and add them back onto the
fitted structure. ``cimba.bootstrap`` provides this family directly; the
factories take the raw series and fit the structure internally:

* ``residual(data, length, trend=1, period=None, mean_block=None)`` -- the
  general residual bootstrap. ``trend`` is a polynomial degree; giving a
  ``period`` switches to a robust STL decomposition whose seasonal component
  tiles beyond the data and whose trend extrapolates linearly. Residuals are
  resampled i.i.d., or with the stationary bootstrap when ``mean_block`` is
  given.
* ``wild(data, trend=1, period=None, weights="rademacher")`` -- the **wild
  bootstrap** for heteroskedastic residuals (variance that changes over
  time): each residual stays at its own time position and is multiplied by a
  random weight (``"rademacher"``, ``"mammen"``, or ``"normal"``) instead of
  being permuted.
* ``sieve(data, length, order=None)`` -- the **sieve bootstrap**: fits an
  AR(p) (order selected by AIC, coefficients by Yule--Walker so the simulated
  process is always stationary), resamples its innovations, and simulates the
  series forward.

.. code-block:: python

   demand = bootstrap.residual(history, length=horizon, period=7,
                               mean_block=14)
   exp = model.experiment(demand=demand, replications=200, seed=42)

``trend`` and ``period`` also accept ``"auto"`` (AICc degree selection and
periodogram-based period detection). Detection is deterministic in the data,
so seed reproducibility is unaffected -- but it is data-dependent, so pin the
values when you know them.

State-dependent dynamics: Markov bootstraps
-------------------------------------------

When the process is better described by "what happens next depends on the
current state" than by a timeline -- regime-switching demand, queue-driven
feedback -- resample transitions conditional on the current state (local or
Markov bootstrap) rather than blocks of the timeline. There is no factory for
this in ``cimba.bootstrap``; write the sampler as an ordinary ``f(rng)``
callable that walks the fitted transition structure and returns the visited
values.

Scarce data: the parametric bootstrap
-------------------------------------

Fit a distribution or time-series model and sample trajectories from it. This
is classical simulation input modeling -- exactly what resampling methods let
you avoid when the data is rich enough that no parametric family is trusted.
It becomes the right tool when data is scarce: a fitted model extrapolates
beyond the observed range, while resampling can only replay what happened.

Other methods
-------------

Worth knowing, more rarely needed: the **Bayesian bootstrap** (random weights
instead of resampling, giving smoother uncertainty), **m-out-of-n
subsampling** (consistent in cases where the standard bootstrap fails, such
as extremes and maxima), the **maximum entropy bootstrap** (non-stationary
series without differencing), and **phase scrambling** (preserves the power
spectrum exactly; more common in physics than in operations research). All of
them fit the same ``f(rng)`` shape if implemented by hand.

Demand data in supply chains
----------------------------

Four hazards come up so often with demand data that they have dedicated
support; the fifth is a data problem no resampler can fix.

**Intermittent demand.** Spare parts and slow movers -- long runs of zeros
with occasional spikes -- break STL and AR fitting, and residual resampling
destroys the zero structure. ``intermittent(data, length)`` follows Willemain
et al.: demand *occurrence* is a two-state Markov chain fitted to the
zero/nonzero pattern (preserving the clustering of demand periods), and each
occurrence draws a *size* from the observed nonzero values. ``jitter=True``
perturbs drawn sizes so values absent from the history can occur.

**Correlated SKUs.** Trace fields deliberately draw independent streams, so
bootstrapping two demand fields separately destroys their cross-correlation --
exactly what stresses shared capacity in a multi-echelon model.
``joint`` resamples a panel of series with *one* set of block choices:

.. code-block:: python

   gens = bootstrap.joint({"demand_a": hist_a, "demand_b": hist_b},
                          length=400, name="demand", mean_block=7)
   exp = model.experiment(**gens, replications=200, seed=42)

Every generator carries ``trace_rng_name = "joint:demand"``, which overrides
the field name in the per-trial rng derivation -- all members receive
identical generators, hence identical block draws, and the historical
cross-correlation survives. Rebuild any trial's draw with
``sim.trace_rng(trial_seed, "joint:demand")``.

**Negative values.** The model-based factories add continuous residuals onto
a fitted structure, so low-volume demand (or an extrapolated downward trend)
can go negative -- and a simulation consuming the trace will not complain.
``nonnegative=True`` on ``residual``, ``wild``, and ``sieve`` clips at zero;
the block methods never need it, since they only replay observed values.

**Future horizons.** By default the fitted structure replays the historical
window (right for input-uncertainty studies). ``start=len(history)`` shifts
the structure window to the horizon *after* the data -- trend extrapolated,
seasonal phase aligned -- for "simulate next year under this trend" studies.

**Stockout censoring.** Sales history records demand *censored by
availability*: every stockout hides the demand that went unserved.
Bootstrapping sales reproduces that censoring, so the simulation
systematically under-stresses the policy being evaluated. No resampler fixes
this -- uncensor upstream (e.g. estimate lost sales from stockout periods) or
read the results as lower bounds on required inventory.

Input uncertainty versus stochastic uncertainty
-----------------------------------------------

Bootstrapping inputs answers a specific question: how much do the model's
conclusions depend on having observed only a finite history? Keep that
distinct from ordinary stochastic uncertainty across replications:

* Replications with **different resamples** (a callable trace, fresh per
  trial) measure input uncertainty plus simulation noise.
* Replications with the **same trace but different seeds** (a shared 1-D
  trace, or repeated rows in a 2-D trace) isolate stochastic uncertainty for
  a fixed input trajectory.

The 2-D trace form supports both in one experiment: a design sweep can hold
the trace fixed while seeds vary, or vary the trace per trial. Trial ordering
rules are in :doc:`traces`.

Writing your own generator
--------------------------

Every method above reduces to a callable that takes a NumPy generator and
returns one 1-D trajectory. Anything expressible in ordinary Python --
fitted statsmodels or scikit-learn models included -- can therefore drive a
trace field, with the experiment seed reproducing every draw:

.. code-block:: python

   def my_generator(rng):
       ...            # any Python + NumPy, seeded only through rng
       return trajectory   # 1-D, length >= warmup + duration + cooldown

Keep all randomness routed through ``rng``: a generator that consults global
random state breaks the reproducibility guarantee that a single experiment
seed otherwise provides. For expensive generators, precompute rows in
parallel with ``model.trial_seeds`` as shown in :doc:`traces`.
