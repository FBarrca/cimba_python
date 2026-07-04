Data-driven Traces
==================

Declare ``demand: sim.Trace`` and pass per-trial replay data to
``model.experiment(demand=...)``: a 1-D array is shared by every trial, a 2-D
array maps row *i* to trial *i* (trial order is design-point-major with
replications innermost), and a sequence of 1-D arrays gives ragged per-trial
traces.

A trace field also accepts a callable ``f(rng)`` or ``f(rng, trial_index)``
returning a 1-D array -- the idiom for bootstrap resampling and fitted
generators. It runs once per trial with a ``numpy.random.Generator`` seeded
from that trial's own cimba seed and the field name, so the single
``experiment(seed=...)`` argument reproduces the simulation streams and the
generated traces together, and distinct trace fields draw independent
streams. ``sim.trace_rng(trial_seed, field_name)`` rebuilds any trial's
generator from its recorded ``exp["seed"]``, e.g. to inspect the trace a
failed trial replayed.

Callable traces run serially inside ``experiment()``, before the parallel
trial run -- negligible for bootstrap resampling, but a bottleneck when a
single generation is expensive (fitted time-series or ML models).
``model.trial_seeds(seed=..., replications=..., **params)`` returns the exact
per-trial seeds that ``experiment()`` will assign, so such traces can be
generated in parallel outside cimba and passed in as finished rows, with
bit-identical results::

   seeds = model.trial_seeds(seed=42, scale=[1.0, 2.0], replications=100)
   rows = Parallel(n_jobs=-1)(
       delayed(slow_generator)(sim.trace_rng(s, "demand")) for s in seeds)
   exp = model.experiment(scale=[1.0, 2.0], demand=rows,
                          replications=100, seed=42)

``cimba.bootstrap`` provides ready-made trace generators that resample an
observed series: ``iid(data, length)`` for serially independent data,
``moving_block(data, length, block)`` and ``circular_block(data, length,
block)`` for stationary dependent series, and ``stationary(data, length,
mean_block)`` (random geometric block lengths -- a good default for
autocorrelated data such as demand histories). Each returns an ``f(rng)``
closure to pass directly as a trace field value::

   from cimba import bootstrap

   demand = bootstrap.stationary(history, length=horizon, mean_block=7)
   exp = model.experiment(demand=demand, replications=200, seed=42)

For trending, seasonal, or autoregressive data there are three model-based
factories that fit the structure internally from the raw series:
``residual(data, length, trend=1, period=None, mean_block=None)`` (polynomial
trend or, with a ``period``, STL decomposition; residuals resampled i.i.d. or
stationary-block), ``wild(data, length=None, trend=1, period=None,
weights="rademacher")`` (heteroskedastic residuals, weighted in place), and
``sieve(data, length, order=None)`` (AR(p) with AIC order selection and
Yule--Walker coefficients, simulated forward with resampled innovations).
``trend`` and ``period`` also accept ``"auto"``; all three take
``nonnegative=True`` (clip at zero, for demand data) and ``start`` (evaluate
the structure on ``start..start+length-1``, e.g. ``start=len(data)`` for the
horizon after the history).

For supply-chain demand there are two more:
``intermittent(data, length, jitter=False)`` (zero-inflated series:
Markov-chain occurrence plus resampled nonzero sizes) and
``joint(panel, length, name=..., mean_block=...)`` (a mapping of field name to
series, resampled with shared block draws so cross-correlation survives;
the returned generators carry a ``trace_rng_name`` attribute, which
``experiment()`` uses instead of the field name when deriving each trial's
generator -- callables sharing the tag receive identical rngs).

Size ``length`` to cover warmup + duration + cooldown.

Inside a process body, ``values = sim.Trace(env.demand)`` returns the
trial's trace as a ``float64`` NumPy view supporting ``len()``, indexing,
slicing, and iteration; treat it as read-only. A generator that exhausts its
trace simply finishes -- the trial still runs to its configured recording
window, so generate traces that cover ``warmup + duration + cooldown`` (or
derive the experiment duration from the trace span), and consider recording
``sim.now()`` into an ``Output`` when the loop ends as an exhaustion check.

For a narrative walkthrough of trace replay and bootstrapping, see
:doc:`../advanced/traces` and :doc:`../advanced/bootstrapping`.
