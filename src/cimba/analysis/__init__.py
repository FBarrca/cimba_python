"""Post-run summaries, paired comparisons and input-model checks."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, cast, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf

from cimba.inputs.sources import DistributionSource, RowSource, TraceSource, trace_rng
from cimba.inputs.bootstrap import JointMembers
from cimba.results import Samples

__all__ = ["Summary", "Comparison", "InputCheck", "JointInputCheck",
           "summary", "compare", "check_input"]


@dataclass(frozen=True)
class Summary:
    n: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def summary(samples: Samples | ArrayLike, *, confidence: float = 0.95) -> Summary:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or not 0 < confidence < 1:
        raise ValueError("summary expects a (points, replications) array and confidence in (0, 1)")
    n = np.isfinite(values).sum(axis=1)
    mean = np.full(values.shape[0], np.nan)
    std = np.full(values.shape[0], np.nan)
    for point in range(values.shape[0]):
        valid = values[point, np.isfinite(values[point])]
        if len(valid):
            mean[point] = valid.mean()
        if len(valid) > 1:
            std[point] = valid.std(ddof=1)
    critical = stats.t.ppf((1 + confidence) / 2, np.maximum(n - 1, 1))
    margin = critical * std / np.sqrt(np.maximum(n, 1))
    return Summary(n, mean, std, mean - margin, mean + margin)


@dataclass(frozen=True)
class Comparison:
    a: int
    b: int
    n: int
    difference: float
    lower: float
    upper: float


def compare(samples: Samples | ArrayLike, *, a: int, b: int,
            confidence: float = 0.95) -> Comparison:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or not 0 < confidence < 1:
        raise ValueError("compare expects a (points, replications) array")
    paired = values[b] - values[a]
    paired = paired[np.isfinite(paired)]
    if len(paired) < 2:
        return Comparison(a, b, len(paired), float("nan"),
                          float("nan"), float("nan"))
    mean = float(paired.mean())
    margin = float(stats.t.ppf((1 + confidence) / 2, len(paired) - 1) *
                   paired.std(ddof=1) / np.sqrt(len(paired)))
    return Comparison(a, b, len(paired), mean, mean - margin, mean + margin)


@dataclass(frozen=True)
class InputCheck:
    reference_mean: float
    generated_mean: float
    reference_variance: float
    generated_variance: float
    reference_quantiles: tuple[float, ...]
    generated_quantiles: tuple[float, ...]
    reference_acf: tuple[float, ...]
    generated_acf: tuple[float, ...]
    reference_pacf: tuple[float, ...]
    generated_pacf: tuple[float, ...]
    ks_statistic: float
    ks_pvalue: float


@dataclass(frozen=True)
class JointInputCheck:
    keys: tuple
    members: tuple[InputCheck, ...]
    reference_correlation: np.ndarray
    generated_correlation: np.ndarray

    def __post_init__(self):
        self.reference_correlation.flags.writeable = False
        self.generated_correlation.flags.writeable = False


@overload
def check_input(source: JointMembers, *, reference: ArrayLike | Mapping,
                n: int = 4096, seed: int = 0, lags: int = 10) -> JointInputCheck: ...


@overload
def check_input(source: RowSource | TraceSource | DistributionSource,
                *, reference: ArrayLike | Mapping, n: int = 4096,
                seed: int = 0, lags: int = 10) -> InputCheck: ...


def check_input(source: RowSource | TraceSource | DistributionSource | JointMembers,
                *, reference: ArrayLike | Mapping, n: int = 4096,
                seed: int = 0, lags: int = 10) -> InputCheck | JointInputCheck:
    if isinstance(source, JointMembers):
        keys = tuple(source)
        if isinstance(reference, Mapping):
            if set(reference) != set(keys):
                raise ValueError("joint reference keys must match the source")
            columns = [np.asarray(reference[key], dtype=np.float64)
                       for key in keys]
            if any(column.ndim != 1 for column in columns):
                raise ValueError("joint reference columns must be 1-D")
            history_matrix = np.column_stack(columns)
        else:
            history_matrix = np.asarray(reference, dtype=np.float64)
        if (history_matrix.ndim != 2 or
            history_matrix.shape[1] != len(keys) or
            history_matrix.shape[0] < 2 or
            not np.isfinite(history_matrix).all()):
            raise ValueError("joint reference must have finite rows and one column per member")
        if n < 2 or lags < 0:
            raise ValueError("n must be >= 2 and lags nonnegative")
        generated_columns = []
        members = []
        for column, key in enumerate(keys):
            member = source[key]
            length = min(n, member.max_length) if member.max_length is not None else n
            generated_columns.append(member.generate(
                [trace_rng(seed, member.tag or source.tag)], length)[0])
            members.append(check_input(
                member, reference=history_matrix[:, column],
                n=n, seed=seed, lags=lags))
        if len({len(column) for column in generated_columns}) != 1:
            raise ValueError("joint members generated unequal row lengths")
        generated_matrix = np.column_stack(generated_columns)
        return JointInputCheck(
            keys, tuple(members),
            np.atleast_2d(np.corrcoef(history_matrix, rowvar=False)),
            np.atleast_2d(np.corrcoef(generated_matrix, rowvar=False)))
    history = np.asarray(reference, dtype=np.float64)
    if history.ndim != 1 or history.size < 2 or not np.isfinite(history).all():
        raise ValueError("reference must be a finite 1-D series with at least two values")
    if n < 2 or lags < 0:
        raise ValueError("n must be >= 2 and lags nonnegative")
    if isinstance(source, TraceSource):
        generated = source.values
    elif isinstance(source, RowSource):
        length = min(n, source.max_length) if source.max_length is not None else n
        generated = source.generate([trace_rng(seed, source.tag or "check_input")],
                                    length)[0]
    else:
        # This diagnostic runs before an experiment; a host RNG is appropriate
        # here and does not affect the native trial stream.
        rng = trace_rng(seed, "check_input")
        params = dict(source.parameters)
        match source.method:
            case "exponential": generated = rng.exponential(params["mean"], n)
            case "normal": generated = rng.normal(params["mean"], params["sd"], n)
            case "gamma": generated = rng.gamma(params["shape"], params["scale"], n)
            case "lognormal": generated = rng.lognormal(params["mean"], params["sigma"], n)
            case "weibull": generated = params["scale"] * rng.weibull(params["shape"], n)
            case "poisson": generated = rng.poisson(params["mean"], n)
            case "triangular": generated = rng.triangular(params["low"], params["mode"], params["high"], n)
            case "pert":
                width = params["high"] - params["low"]
                alpha = 1 + 4 * (params["mode"] - params["low"]) / width
                beta = 1 + 4 * (params["high"] - params["mode"]) / width
                generated = params["low"] + width * rng.beta(alpha, beta, n)
            case "categorical": generated = rng.choice(params["values"], n, p=params["probabilities"])
            case "uniform": generated = rng.uniform(params["low"], params["high"], n)
            case "logistic": generated = rng.logistic(params["mean"], params["scale"], n)
            case "cauchy": generated = params["mode"] + params["scale"] * rng.standard_cauchy(n)
            case "erlang": generated = rng.gamma(params["k"], params["mean"] / params["k"], n)
            case "beta": generated = (params["low"] + (params["high"] - params["low"]) *
                                      rng.beta(params["a"], params["b"], n))
            case "pert_mod":
                width = params["high"] - params["low"]
                alpha = 1 + params["weight"] * (params["mode"] - params["low"]) / width
                beta = 1 + params["weight"] * (params["high"] - params["mode"]) / width
                generated = params["low"] + width * rng.beta(alpha, beta, n)
            case "rayleigh": generated = rng.rayleigh(params["scale"], n)
            case "bernoulli": generated = rng.binomial(1, params["p"], n)
            case "geometric": generated = rng.geometric(params["p"], n)
            case "binomial": generated = rng.binomial(params["n"], params["p"], n)
            case "negative_binomial":
                generated = rng.negative_binomial(params["successes"], params["p"], n)
            case "pareto": generated = params["mode"] * (rng.pareto(params["shape"], n) + 1)
            case "chi_squared": generated = rng.chisquare(params["df"], n)
            case "f_dist": generated = rng.f(params["df1"], params["df2"], n)
            case "student_t":
                generated = params["mean"] + params["scale"] * rng.standard_t(params["df"], n)
            case "dice": generated = rng.integers(params["low"], params["high"] + 1, n)
            case "hypoexponential":
                generated = sum(rng.exponential(mean, n) for mean in params["means"])
            case "hyperexponential":
                choice = rng.choice(len(params["means"]), n, p=params["probabilities"])
                generated = rng.exponential(np.asarray(params["means"])[choice])
            case _: raise ValueError(f"check_input does not support {source.method}")
    generated = np.asarray(generated, dtype=np.float64)
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    nlags = min(lags, len(history) - 1, len(generated) - 1)
    pacf_lags = min(nlags, max(0, len(history) // 2 - 1),
                    max(0, len(generated) // 2 - 1))
    ks = cast(Any, stats.ks_2samp(history, generated))
    return InputCheck(
        float(history.mean()), float(generated.mean()),
        float(history.var(ddof=1)), float(generated.var(ddof=1)),
        tuple(np.quantile(history, quantiles).tolist()),
        tuple(np.quantile(generated, quantiles).tolist()),
        tuple(np.asarray(acf(history, nlags=nlags, fft=True)).tolist()),
        tuple(np.asarray(acf(generated, nlags=nlags, fft=True)).tolist()),
        tuple(np.asarray(pacf(history, nlags=pacf_lags, method="ywm")).tolist()),
        tuple(np.asarray(pacf(generated, nlags=pacf_lags, method="ywm")).tolist()),
        float(ks.statistic), float(ks.pvalue),
    )
