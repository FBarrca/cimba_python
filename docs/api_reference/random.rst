Random Draws
============

Each trial receives its own pseudo-random stream. Passing ``seed=...`` to
``model.experiment(...)`` makes runs reproducible; omitting it lets the package
choose independent seeds. Call these inside process bodies like any other
``sim`` function, and declare distribution parameters as ``sim.Param`` fields to
sweep them across the experiment table.

``exponential()``, ``gamma()``, ``uniform()``, ``normal()``, ``random01()``,
``rayleigh()``, ``pert()``, ``pert_mod()``, ``bernoulli()``, ``flip()``,
``triangular()``, ``weibull()``, ``lognormal()``, ``erlang()``, ``beta()``,
``poisson()``, ``dice()``, ``std_normal()``, ``std_exponential()``,
``std_gamma()``, ``std_beta()``, ``logistic()``, ``cauchy()``, ``pareto()``,
``chisquared()``, ``f_dist()``, ``std_t()``, ``t_dist()``, ``geometric()``,
``binomial()``, ``negative_binomial()``, ``pascal()``, ``hypoexponential()``,
``hyperexponential()``, ``categorical()``, ``loaded_dice()``.
