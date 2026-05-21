# This file is included by ../_cimba.pyx.

def hwseed() -> int:
    """Return a hardware-derived random seed."""
    return <object>cmb_random_hwseed()


def seed(object value=None) -> int:
    """Initialize the thread-local random generator and return the seed used."""
    cdef uint64_t seed_value = cmb_random_hwseed() if value is None else <uint64_t>value
    cmb_random_initialize(seed_value)
    return <object>seed_value


def current_seed() -> int:
    """Return the seed currently used by Cimba in this thread."""
    return <object>cmb_random_curseed()


def random() -> float:
    return cmb_random()


def random_u64() -> int:
    return <object>cmb_random_sfc64()


def fmix64(int seed, int nonce) -> int:
    return <object>cmb_random_fmix64(<uint64_t>seed, <uint64_t>nonce)


def uniform(double min, double max) -> float:
    return cmb_random_uniform(min, max)


def triangular(double min, double mode, double max) -> float:
    return cmb_random_triangular(min, mode, max)


def normal(double mu=0.0, double sigma=1.0) -> float:
    return cmb_random_normal(mu, sigma)


def exponential(double mean) -> float:
    return cmb_random_exponential(mean)


def gamma(double shape, double scale=1.0) -> float:
    return cmb_random_gamma(shape, scale)


def beta(double a, double b, double min=0.0, double max=1.0) -> float:
    return cmb_random_beta(a, b, min, max)


def pert(double min, double mode, double max) -> float:
    return cmb_random_PERT(min, mode, max)


def pert_mod(double min, double mode, double max, double lambda_) -> float:
    return cmb_random_PERT_mod(min, mode, max, lambda_)


def dice(int min, int max) -> int:
    return cmb_random_dice(min, max)


def flip() -> bool:
    return True if cmb_random_flip() else False


def bernoulli(double p) -> bool:
    return True if cmb_random_bernoulli(p) else False
