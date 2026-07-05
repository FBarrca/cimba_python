"""Bootstrap trajectory generators for data-driven traces.

Each factory takes an observed 1-D series and returns a callable
``f(rng)`` that produces one resampled trajectory of ``length`` float64
values -- ready to pass as a ``sim.Trace`` field value to
``Model.experiment()``, which invokes it once per trial with that
trial's ``sim.trace_rng()`` generator, so a single experiment ``seed``
reproduces every resample::

    from cimba import bootstrap

    demand = bootstrap.stationary(history, length=horizon, mean_block=7)
    exp = model.experiment(demand=demand, replications=200, seed=42)

Choosing a method:

* ``iid`` -- observations are independent (service times, order sizes).
  Destroys autocorrelation, so it is wrong for serially dependent data.
* ``moving_block`` / ``circular_block`` -- stationary dependent series;
  fixed-length contiguous blocks preserve within-block dependence.
  The circular variant wraps around the end of the series so edge
  observations are not underweighted.
* ``stationary`` -- like the block methods but with random geometric
  block lengths (Politis & Romano), so the resampled series is itself
  stationary rather than having seams every fixed ``block`` steps.
  A good default for autocorrelated data such as demand histories.
* ``residual`` -- trending or seasonal data: fits the deterministic
  structure (polynomial trend and, given a ``period``, STL
  seasonality), resamples the residuals (i.i.d. or stationary-block),
  and adds them back onto the extrapolated structure.
* ``wild`` -- heteroskedastic residuals: keeps each residual at its own
  time position and multiplies it by a random weight, preserving
  variance that changes over time.
* ``sieve`` -- autoregressive dynamics: fits an AR(p) (order selected
  by AIC) and simulates forward with resampled innovations.
* ``intermittent`` -- zero-inflated series (spare parts, slow movers):
  Markov-chain demand occurrence plus resampled nonzero sizes.
* ``joint`` -- several correlated series (related SKUs): one stationary
  resample drives every series, preserving cross-correlation.

``residual``, ``wild``, and ``sieve`` use statsmodels for STL and AR
fitting. Their ``trend``/``period`` arguments also accept ``"auto"``;
``start`` shifts the structure window past the history and
``nonnegative=True`` clips at zero for demand data.

Block length trades off dependence preservation (longer) against
resample diversity (shorter); n**(1/3) is a common starting point.
The pure block factories assume stationarity.

Size ``length`` to cover the experiment's warmup + duration + cooldown:
a generator that exhausts its trace simply finishes while the trial
runs on.
"""

from ._common import TraceGenerator
from .blocks import circular_block, iid, moving_block, stationary
from .demand import intermittent
from .model import residual, sieve, wild
from .panel import joint

__all__ = ["TraceGenerator", "iid", "moving_block", "circular_block",
           "stationary", "residual", "wild", "sieve", "intermittent",
           "joint"]
