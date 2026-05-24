"""Typed surface for Cimba's cmb_random module."""

from collections.abc import Iterable

class AliasSampler:
    """Reusable Vose alias sampler for non-uniform discrete probabilities."""

    def __init__(self, probabilities: Iterable[float]) -> None:
        """Create an alias table from probabilities that sum to 1.0."""
        ...

    def __enter__(self) -> AliasSampler:
        """Enter a context manager and return this sampler."""
        ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        """Close the sampler when leaving a context manager."""
        ...

    def __len__(self) -> int:
        """Return the number of probabilities in the alias table."""
        ...

    def sample(self) -> int:
        """Draw one index according to the configured probabilities."""
        ...

    def close(self) -> None:
        """Destroy the native alias table."""
        ...

def hwseed() -> int:
    """Return a hardware-derived 64-bit seed suitable for Cimba's PRNG."""
    ...

def seed(value: int | None = None) -> int:
    """Initialize the thread-local PRNG and return the seed that was used."""
    ...

def current_seed() -> int:
    """Return the seed used for the current thread's PRNG stream."""
    ...

def random() -> float:
    """Draw a continuous uniform random variate on [0.0, 1.0]."""
    ...

def random_u64() -> int:
    """Draw a raw uniformly distributed 64-bit pseudo-random bit pattern."""
    ...

def fmix64(seed: int, nonce: int) -> int:
    """Mix a master seed and deterministic nonce into a reproducible 64-bit seed."""
    ...

def uniform(min: float, max: float) -> float:
    """Draw from a continuous uniform distribution on [min, max]."""
    ...

def triangular(min: float, mode: float, max: float) -> float:
    """Draw from a triangular distribution with endpoints and a peak mode."""
    ...

def normal(mu: float = 0.0, sigma: float = 1.0) -> float:
    """Draw from a normal distribution with mean mu and standard deviation sigma."""
    ...

def lognormal(m: float, s: float) -> float:
    """Draw from a lognormal distribution with normal parameters m and s."""
    ...

def logistic(m: float, s: float) -> float:
    """Draw from a logistic distribution with location m and scale s."""
    ...

def cauchy(mode: float, scale: float) -> float:
    """Draw from a Cauchy distribution with mode and scale."""
    ...

def exponential(mean: float) -> float:
    """Draw from an exponential distribution with the given mean."""
    ...

def erlang(k: int, mean: float) -> float:
    """Draw from an Erlang distribution with k exponential phases."""
    ...

def hypoexponential(means: Iterable[float]) -> float:
    """Draw from a sum of exponential distributions with the given means."""
    ...

def hyperexponential(means: Iterable[float], probabilities: Iterable[float]) -> float:
    """Draw from one of several exponential distributions by probability."""
    ...

def gamma(shape: float, scale: float = 1.0) -> float:
    """Draw from a gamma distribution with shape and scale parameters."""
    ...

def beta(a: float, b: float, min: float = 0.0, max: float = 1.0) -> float:
    """Draw from a beta distribution scaled to [min, max]."""
    ...

def pert(min: float, mode: float, max: float) -> float:
    """Draw from the standard PERT empirical distribution."""
    ...

def pert_mod(min: float, mode: float, max: float, lambda_: float) -> float:
    """Draw from a modified PERT distribution with an explicit lambda shape."""
    ...

def weibull(shape: float, scale: float) -> float:
    """Draw from a Weibull distribution with shape and scale."""
    ...

def pareto(shape: float, mode: float) -> float:
    """Draw from a Pareto distribution with shape and mode."""
    ...

def chi_squared(k: float) -> float:
    """Draw from a chi-squared distribution with k degrees of freedom."""
    ...

def f_dist(a: float, b: float) -> float:
    """Draw from an F distribution."""
    ...

def student_t(v: float, m: float = 0.0, s: float = 1.0) -> float:
    """Draw from a Student-t distribution with df v, location m, and scale s."""
    ...

def rayleigh(s: float) -> float:
    """Draw from a Rayleigh distribution with scale s."""
    ...

def dice(min: int, max: int) -> int:
    """Draw an integer uniformly from the inclusive interval [min, max]."""
    ...

def flip() -> bool:
    """Draw an unbiased Bernoulli trial."""
    ...

def bernoulli(p: float) -> bool:
    """Draw a Bernoulli trial that is true with probability p."""
    ...

def geometric(p: float) -> int:
    """Draw from a geometric distribution on [1, infinity)."""
    ...

def binomial(n: int, p: float) -> int:
    """Draw from a binomial distribution."""
    ...

def negative_binomial(m: int, p: float) -> int:
    """Draw from a negative binomial distribution."""
    ...

def pascal(m: int, p: float) -> int:
    """Draw from a Pascal distribution."""
    ...

def poisson(r: float) -> int:
    """Draw from a Poisson distribution with rate r."""
    ...

def loaded_dice(probabilities: Iterable[float]) -> int:
    """Draw one index from a non-uniform discrete distribution."""
    ...
