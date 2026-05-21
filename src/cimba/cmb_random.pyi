"""Typed surface for Cimba's cmb_random module."""

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

def exponential(mean: float) -> float:
    """Draw from an exponential distribution with the given mean."""
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

def dice(min: int, max: int) -> int:
    """Draw an integer uniformly from the inclusive interval [min, max]."""
    ...

def flip() -> bool:
    """Draw an unbiased Bernoulli trial."""
    ...

def bernoulli(p: float) -> bool:
    """Draw a Bernoulli trial that is true with probability p."""
    ...
