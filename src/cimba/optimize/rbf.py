"""RBFOpt: a radial-basis-function surrogate search, one candidate at a time.

Optional: needs the ``rbfopt`` package (``pip install cimba[rbf]``). With
``local_search="solver"`` it also needs the Bonmin and Ipopt solvers.
"""

from __future__ import annotations

import os
import shutil
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np
from scipy.stats import qmc

from .de import SearchStopped

ALGORITHMS = ('MSRSM', 'Gutmann')
RBFS = ('auto', 'cubic', 'thin_plate_spline', 'linear', 'multiquadric', 'gaussian')
LOCAL_SEARCHES = ('surrogate', 'solver')


def _solver(name, path):
    """``path``, else ``name`` on PATH, else the AMPL ``ampl_module_coin`` build."""
    if path is not None:
        return path
    if found := shutil.which(name):
        return found
    try:
        import ampl_module_coin
    except ImportError:
        raise RuntimeError(
            f"RBFOpt needs the {name} solver: put it on PATH, pass its path, or "
            "pip install --extra-index-url https://pypi.ampl.com ampl-module-coin") from None
    return os.path.join(ampl_module_coin.bin_dir, name)


@contextmanager
def _surrogate_local_search():
    """Minimize the surrogate with RBFOpt's own genetic search, in process.

    RBFOpt's local step otherwise writes the surrogate to a file and runs
    Bonmin on it, which costs seconds per proposal; the surrogate itself is
    cheap to evaluate on thousands of points at once.
    """
    import rbfopt.rbfopt_aux_problems as aux
    import rbfopt.rbfopt_utils as ru

    def minimize_rbf(settings, n, k, var_lower, var_upper, integer_vars, categorical_info,
                     node_pos, rbf_lambda, rbf_h, best_node_pos):
        def surrogate(points):
            return ru.bulk_evaluate_rbf(settings, np.atleast_2d(points), n, k,
                                        node_pos, rbf_lambda, rbf_h)
        point = aux.ga_optimize(settings, n, var_lower, var_upper, integer_vars,
                                categorical_info, surrogate)
        best = np.array(best_node_pos, dtype=float)
        return point if surrogate(point)[0] < surrogate(best)[0] else best

    original = aux.minimize_rbf
    aux.minimize_rbf = minimize_rbf
    try:
        yield
    finally:
        aux.minimize_rbf = original


@dataclass(frozen=True, kw_only=True)
class RBFOpt:
    initial: int | None = None
    max_evaluations: int = 300
    max_iterations: int = 1000
    algorithm: str = 'MSRSM'
    rbf: str = 'auto'
    local_search: str = 'surrogate'
    minlp_solver_path: str | None = None
    nlp_solver_path: str | None = None

    def __post_init__(self):
        for name, minimum in (('initial', 2), ('max_evaluations', 1), ('max_iterations', 1)):
            value = getattr(self, name)
            if name == 'initial' and value is None:
                continue
            if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {', '.join(ALGORITHMS)}")
        if self.rbf not in RBFS:
            raise ValueError(f"rbf must be one of {', '.join(RBFS)}")
        if self.local_search not in LOCAL_SEARCHES:
            raise ValueError(f"local_search must be one of {', '.join(LOCAL_SEARCHES)}")

    def resolved(self, dimensions, replications, workers):
        settings = asdict(self)
        if self.initial is None:
            settings['initial'] = 2 * dimensions + 1
        return settings

    def population_size(self, dimensions, settings):
        return settings['initial']

    def generation_size(self, dimensions, settings):
        # After the initial batch, RBFOpt proposes one candidate at a time.
        return 1

    def search(self, space, evaluate, rng, *, settings, initial=None, callback=None):
        with warnings.catch_warnings():
            # rbfopt 4.3's docstrings have invalid escapes.
            warnings.simplefilter('ignore', SyntaxWarning)
            import rbfopt
        bounds = np.asarray(space.bounds, dtype=float)
        integral = np.asarray(space.integrality)
        seen = []
        evaluated = 0
        # An all-integer space has finitely many points. RBFOpt keeps
        # proposing known ones once all are simulated, so stop there.
        simulated = set()
        size = np.prod(bounds[:, 1] - bounds[:, 0] + 1) if integral.all() else np.inf

        class Exhausted(Exception):
            pass

        def energies(points):
            nonlocal evaluated
            values = evaluate(points)
            evaluated += len(points)
            if integral.all():
                simulated.update(map(tuple, np.rint(points)))
            finite = np.isfinite(values)
            seen.extend(values[finite])
            if callback is not None and callback(None):
                raise SearchStopped()
            if len(simulated) >= size:
                raise Exhausted()
            # RBFOpt can't fit infinite values, so rejected candidates score
            # well above the worst seen.
            return np.where(finite, values, 2 * abs(max(seen, default=1.0)) + 1)

        unit = qmc.LatinHypercube(d=len(bounds), rng=rng).random(settings['initial'])
        nodes = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
        if initial is not None:
            starts = [space.encode(values) for values in initial]
            if len(starts) > len(nodes):
                raise ValueError("initial has more points than RBFOpt's initial batch")
            nodes[:len(starts)] = starts
        nodes[:, integral] = np.rint(nodes[:, integral])
        # Drop repeats, keeping the starting points first.
        nodes = nodes[np.sort(np.unique(nodes, axis=0, return_index=True)[1])]
        black_box = rbfopt.RbfoptUserBlackBox(
            len(bounds), bounds[:, 0], bounds[:, 1], np.where(integral, 'I', 'R'),
            lambda x: float(energies(np.asarray([x]))[0]))
        options = rbfopt.RbfoptSettings(
            # RBFOpt's own count excludes the initial batch.
            max_evaluations=max(1, self.max_evaluations - len(nodes)),
            max_iterations=self.max_iterations, algorithm=self.algorithm, rbf=self.rbf,
            rand_seed=int(rng.integers(1 << 31)), num_cpus=1, **self._solvers())
        # RBFOpt seeds and draws from NumPy's global generator; restore it.
        state = np.random.get_state()
        local = _surrogate_local_search() if self.local_search == 'surrogate' else nullcontext()
        try:
            with local, open(os.devnull, 'w') as quiet:
                algorithm = rbfopt.RbfoptAlgorithm(
                    options, black_box, init_node_pos=nodes, init_node_val=energies(nodes),
                    do_init_strategy=False)
                algorithm.set_output_stream(quiet)
                _, _, iterations, _, _ = algorithm.optimize()
        except SearchStopped:
            return 'budget'
        except Exhausted:
            return 'converged'
        finally:
            np.random.set_state(state)
        if iterations >= self.max_iterations or evaluated >= self.max_evaluations:
            return 'maxiter'
        # RBFOpt's own stopping rules hold.
        return 'converged'

    def _solvers(self):
        if self.local_search == 'surrogate':
            # Not run: the surrogate local search and RBFOpt's genetic global
            # search need no solver, and the black box is never noisy.
            return {}
        return {'minlp_solver_path': _solver('bonmin', self.minlp_solver_path),
                'nlp_solver_path': _solver('ipopt', self.nlp_solver_path)}
