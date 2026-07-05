"""AST lowering for compiled ``cimba.random`` calls."""

from __future__ import annotations

import ast
import copy
import inspect
import linecache
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import FunctionType
from typing import Any

from . import _compiled
from . import __name__ as _RANDOM_MODULE_NAME


@dataclass(frozen=True)
class _RandomFunctionSpec:
    helper_name: str
    helper_attr: str
    params: tuple[str, ...] = ()
    defaults: Mapping[str, object] = field(default_factory=dict)

    def normalize_args(
        self,
        function: str,
        args: Sequence[ast.expr],
        keywords: Sequence[ast.keyword],
        *,
        label: str,
    ) -> list[ast.expr]:
        if len(args) > len(self.params):
            raise ValueError(
                f"{label} passes too many arguments to random.{function}()")
        call_args = list(args)
        if not keywords:
            return self._fill_defaults(function, call_args, label=label)

        by_name = {name: index for index, name in enumerate(self.params)}
        supplied = set(self.params[:len(call_args)])
        keyed: dict[int, ast.expr] = {}
        max_index = len(call_args) - 1
        for kw in keywords:
            if kw.arg is None:
                raise ValueError(
                    f"{label} cannot use **kwargs with random.{function}()")
            index = by_name.get(kw.arg)
            if index is None:
                raise ValueError(
                    f"{label} passes unknown random.{function}() argument "
                    f"'{kw.arg}'")
            if kw.arg in supplied or index in keyed:
                raise ValueError(
                    f"{label} passes random.{function}() argument "
                    f"'{kw.arg}' more than once")
            supplied.add(kw.arg)
            keyed[index] = kw.value
            max_index = max(max_index, index)

        for index in range(len(call_args), max_index + 1):
            if index in keyed:
                call_args.append(keyed[index])
            else:
                param = self.params[index]
                if param not in self.defaults:
                    raise ValueError(
                        f"{label} is missing required random.{function}() "
                        f"argument '{param}'")
                call_args.append(ast.Constant(self.defaults[param]))
        return self._fill_defaults(function, call_args, label=label)

    def _fill_defaults(
        self,
        function: str,
        call_args: list[ast.expr],
        *,
        label: str,
    ) -> list[ast.expr]:
        for index in range(len(call_args), len(self.params)):
            param = self.params[index]
            if param not in self.defaults:
                raise ValueError(
                    f"{label} is missing required random.{function}() "
                    f"argument '{param}'")
            call_args.append(ast.Constant(self.defaults[param]))
        return call_args


def _spec(
    function: str,
    params: tuple[str, ...],
    *,
    helper_attr: str | None = None,
    defaults: Mapping[str, object] | None = None,
) -> _RandomFunctionSpec:
    attr = helper_attr or function
    return _RandomFunctionSpec(
        helper_name=f"_cimba_random_{attr}",
        helper_attr=attr,
        params=params,
        defaults=defaults or {},
    )


_FUNCTION_SPECS = {
    "uniform": _spec(
        "uniform", ("min", "max"), defaults={"min": 0.0, "max": 1.0}),
    "exponential": _spec(
        "exponential", ("mean",), defaults={"mean": 1.0}),
    "gamma": _spec("gamma", ("shape", "scale"), defaults={"scale": 1.0}),
    "normal": _spec(
        "normal", ("mu", "sigma"), defaults={"mu": 0.0, "sigma": 1.0}),
    "rayleigh": _spec("rayleigh", ("s",)),
    "pert": _spec("pert", ("min", "mode", "max")),
    "pert_mod": _spec("pert_mod", ("min", "mode", "max", "lambda_")),
    "bernoulli": _spec("bernoulli", ("p",)),
    "triangular": _spec("triangular", ("min", "mode", "max")),
    "weibull": _spec("weibull", ("shape", "scale")),
    "lognormal": _spec("lognormal", ("m", "s")),
    "erlang": _spec("erlang", ("k", "mean")),
    "beta": _spec(
        "beta",
        ("a", "b", "min", "max"),
        defaults={"min": 0.0, "max": 1.0},
    ),
    "poisson": _spec("poisson", ("r",)),
    "dice": _spec("dice", ("min", "max")),
    "logistic": _spec("logistic", ("m", "s")),
    "cauchy": _spec("cauchy", ("mode", "scale")),
    "pareto": _spec("pareto", ("shape", "mode")),
    "chi_squared": _spec("chi_squared", ("k",)),
    "f_dist": _spec("f_dist", ("a", "b")),
    "student_t": _spec(
        "student_t",
        ("v", "m", "s"),
        defaults={"m": 0.0, "s": 1.0},
    ),
    "geometric": _spec("geometric", ("p",)),
    "binomial": _spec("binomial", ("n", "p")),
    "negative_binomial": _spec("negative_binomial", ("m", "p")),
    "hypoexponential": _spec("hypoexponential", ("means",)),
    "hyperexponential": _spec(
        "hyperexponential", ("means", "probabilities")),
    "categorical": _spec("categorical", ("probabilities",)),
}

RANDOM_FUNCTION_NAMES = frozenset(_FUNCTION_SPECS)


def random_lowering_namespace() -> dict[str, Any]:
    return {
        spec.helper_name: getattr(_compiled, spec.helper_attr)
        for spec in _FUNCTION_SPECS.values()
    }


def _visit_expr(
    visit: Callable[[ast.AST], ast.AST],
    node: ast.expr,
    *,
    what: str,
) -> ast.expr:
    lowered = visit(node)
    if not isinstance(lowered, ast.expr):
        raise TypeError(f"{what} did not lower to an expression")
    return lowered


def _namespace_names(namespace: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    random_names: set[str] = set()
    random_owner_names: set[str] = set()
    for name, value in namespace.items():
        if getattr(value, "__name__", None) == _RANDOM_MODULE_NAME:
            random_names.add(name)
        elif getattr(getattr(value, "random", None), "__name__", None) \
                == _RANDOM_MODULE_NAME:
            random_owner_names.add(name)
    return random_names, random_owner_names


class _RandomCallLowerer(ast.NodeTransformer):
    def __init__(
        self,
        *,
        random_names: set[str],
        random_owner_names: set[str],
        label: str,
    ):
        self.random_names = random_names
        self.random_owner_names = random_owner_names
        self.label = label
        self.changed = False

    def _random_function(self, node: ast.AST) -> str | None:
        if not isinstance(node, ast.Attribute):
            return None
        if isinstance(node.value, ast.Name) and node.value.id in self.random_names:
            return node.attr
        if (isinstance(node.value, ast.Attribute)
                and node.value.attr == "random"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in self.random_owner_names):
            return node.attr
        return None

    def visit_Call(self, node: ast.Call) -> ast.AST:
        function = self._random_function(node.func)
        if function is None:
            return self.generic_visit(node)

        spec = _FUNCTION_SPECS.get(function)
        if spec is None:
            raise ValueError(
                f"{self.label} uses unsupported random function "
                f"random.{function}()")
        self.changed = True
        args = [
            _visit_expr(self.visit, arg, what="random function argument")
            for arg in node.args
        ]
        keywords = [
            ast.keyword(
                arg=kw.arg,
                value=_visit_expr(
                    self.visit, kw.value, what="random function keyword"),
            )
            for kw in node.keywords
        ]
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=spec.helper_name, ctx=ast.Load()),
                args=spec.normalize_args(
                    function, args, keywords, label=self.label),
                keywords=[],
            ),
            node,
        )


def lower_random_calls_in_node(
    node: ast.FunctionDef,
    *,
    namespace: Mapping[str, Any],
    label: str,
) -> tuple[ast.FunctionDef, bool]:
    random_names, random_owner_names = _namespace_names(namespace)
    if not random_names and not random_owner_names:
        return node, False

    lowerer = _RandomCallLowerer(
        random_names=random_names,
        random_owner_names=random_owner_names,
        label=label,
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError("random call lowering produced a non-function")
    return lowered, lowerer.changed


def lower_random_calls_in_function(
    fn: Callable[..., Any],
    *,
    label: str,
) -> Callable[..., Any]:
    namespace = _closure_namespace(fn)
    random_names, random_owner_names = _namespace_names(namespace)
    names = set(fn.__code__.co_names)
    if not (names.intersection(RANDOM_FUNCTION_NAMES)
            and names.intersection(random_names | random_owner_names)):
        return fn

    try:
        node = copy.deepcopy(_function_def_from_source(fn))
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"{label} needs inspectable source to use cimba.random"
        ) from exc

    lowered, changed = lower_random_calls_in_node(
        node, namespace=namespace, label=label)
    if not changed:
        return fn

    _strip_function_metadata(lowered)
    namespace.update(random_lowering_namespace())
    return _compile_lowered(
        lowered,
        filename=f"<cimba random callback '{fn.__qualname__}'>",
        fn_name=fn.__name__,
        qualname=fn.__qualname__,
        namespace=namespace,
        like=fn,
    )


def _strip_function_metadata(node: ast.FunctionDef) -> None:
    node.decorator_list = []
    node.returns = None
    node.type_comment = None
    for arg in node.args.args:
        arg.annotation = None
        arg.type_comment = None


def _closure_namespace(fn: Callable[..., Any]) -> dict[str, Any]:
    namespace = dict(fn.__globals__)
    closure = fn.__closure__ or ()
    for name, cell in zip(fn.__code__.co_freevars, closure, strict=True):
        try:
            namespace[name] = cell.cell_contents
        except ValueError:
            pass
    return namespace


def _function_source(fn: Callable[..., Any]) -> str:
    source = getattr(fn, "__cimba_source__", None)
    if source is None:
        source = inspect.getsource(fn)
    return textwrap.dedent(source)


def _function_def_from_source(fn: Callable[..., Any]) -> ast.FunctionDef:
    tree = ast.parse(_function_source(fn))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise TypeError(f"{fn.__qualname__} source does not contain a function")


def _compile_lowered(
    node: ast.FunctionDef,
    *,
    filename: str,
    fn_name: str,
    qualname: str,
    namespace: dict[str, Any],
    like: Callable[..., Any],
) -> Callable[..., Any]:
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    source = ast.unparse(module) + "\n"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(True),
        filename,
    )
    exec(compile(source, filename, "exec"), namespace)
    generated = namespace[fn_name]
    if not isinstance(generated, FunctionType):
        raise TypeError(f"lowered {qualname} did not produce a function")
    generated.__name__ = like.__name__
    generated.__qualname__ = qualname
    generated.__doc__ = like.__doc__
    generated.__module__ = like.__module__
    generated.__cimba_source__ = source
    return generated
