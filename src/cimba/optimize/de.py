"""Differential evolution: one vectorized SciPy evaluation per generation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil, isfinite
from numbers import Integral, Real
from typing import Any, cast

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc


class SearchStopped(Exception):
    """Internal stop before the initial population's first generation."""


@dataclass(frozen=True, kw_only=True)
class DifferentialEvolution:
    popsize: int | str = 15
    mutation: float | tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    strategy: str = "best1bin"
    tol: float = 0.01
    atol: float = 0.0
    maxiter: int = 1000
    init: str = "latinhypercube"

    def __post_init__(self):
        if self.popsize != "fill" and (not isinstance(self.popsize, Integral) or isinstance(self.popsize, bool) or float(self.popsize) < 1):
            raise ValueError("popsize must be a positive integer or 'fill'")
        mutation = (self.mutation,) if isinstance(self.mutation, Real) else self.mutation
        if (not isinstance(mutation, (tuple, list)) or len(mutation) not in (1, 2) or
                any(not isinstance(x, Real) or not isfinite(x) or not 0 <= float(x) < 2 for x in mutation)):
            raise ValueError("mutation must be in [0, 2), or a pair in that range")
        if len(mutation) == 2:
            if mutation[0] >= mutation[1]:
                raise ValueError("mutation pair must be increasing")
            object.__setattr__(self, "mutation", tuple(mutation))
        elif not isinstance(self.mutation, Real):
            raise ValueError("mutation must be a scalar or a pair")
        if not isinstance(self.recombination, Real) or not isfinite(self.recombination) or not 0 <= self.recombination <= 1:
            raise ValueError("recombination must be in [0, 1]")
        strategies = {name + crossover for name in ('best1', 'best2', 'rand1', 'rand2', 'randtobest1', 'currenttobest1')
                      for crossover in ('bin', 'exp')}
        if self.strategy not in strategies:
            raise ValueError(f"unknown DE strategy: {self.strategy}")
        for name in ('tol', 'atol'):
            value = getattr(self, name)
            if not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.maxiter, Integral) or isinstance(self.maxiter, bool) or self.maxiter < 0:
            raise ValueError("maxiter must be a nonnegative integer")
        if self.init not in {'latinhypercube', 'random', 'sobol', 'halton'}:
            raise ValueError("init must be latinhypercube, random, sobol or halton")

    def resolved(self, dimensions, replications, workers):
        settings = asdict(self)
        if self.popsize == "fill":
            settings['popsize'] = max(15, ceil(256 * workers / (dimensions * replications)))
        return settings

    def population_size(self, dimensions, popsize):
        # rand2 needs five distinct donors in addition to the target member.
        minimum = 6 if self.strategy.startswith('rand2') else 5
        size = max(minimum, dimensions * popsize)
        return 1 << (size - 1).bit_length() if self.init == 'sobol' else size

    def population(self, space, rng, settings, initial):
        size = self.population_size(len(space.dimensions), settings['popsize'])
        if self.init == 'random':
            unit = rng.uniform(size=(size, len(space.dimensions)))
        elif self.init == 'latinhypercube':
            # SciPy's stratified initialization, using only the supplied RNG.
            samples = (rng.uniform(size=(size, len(space.dimensions))) + np.arange(size)[:, None]) / size
            unit = np.column_stack([samples[rng.permutation(size), i] for i in range(len(space.dimensions))])
        else:
            sampler = qmc.Sobol if self.init == 'sobol' else qmc.Halton
            unit = sampler(len(space.dimensions), rng=rng).random(size)
        bounds = np.asarray(space.bounds)
        population = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
        if initial is not None:
            initial = tuple(initial)
            if len(initial) > size:
                raise ValueError("initial has more points than the DE population")
            for i, values in enumerate(initial):
                population[i] = space.encode(values)
        return population

    def search(self, space, evaluate, rng, *, settings, initial=None, callback=None):
        population = self.population(space, rng, settings, initial)
        options = {key: value for key, value in settings.items() if key != 'init'}
        try:
            result = differential_evolution(
                lambda vectors: evaluate(vectors.T), space.bounds,
                integrality=space.integrality, vectorized=True, updating='deferred',
                workers=1, polish=False, rng=rng, callback=callback, disp=False,
                init=cast(Any, population), **options)
        except SearchStopped:
            return 'budget'
        return 'converged' if result.success else 'maxiter'
