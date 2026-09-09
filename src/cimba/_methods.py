"""Shared argument handling for compiled entity and statistics methods."""

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class _MethodSpec:
    helper_name: str
    helper_attr: str
    params: tuple[str, ...] = ()
    defaults: Mapping[str, object] = field(default_factory=dict)
    # Conditions and scheduled events pass the caller's environment implicitly.
    needs_env: bool = False

    def normalize_args(
        self,
        kind: str,
        method: str,
        args: Sequence[ast.expr],
        keywords: Sequence[ast.keyword],
        *,
        label: str,
    ) -> list[ast.expr]:
        call = f"{kind} {method}()"
        if len(args) > len(self.params):
            raise ValueError(f"{label} passes too many arguments to {call}")
        supplied = dict(zip(self.params, args))
        for kw in keywords:
            if kw.arg is None:
                raise ValueError(f"{label} cannot use **kwargs with {call}")
            if kw.arg not in self.params:
                raise ValueError(
                    f"{label} passes unknown {call} argument '{kw.arg}'")
            if kw.arg in supplied:
                raise ValueError(
                    f"{label} passes {call} argument '{kw.arg}' more than once")
            supplied[kw.arg] = kw.value

        # Native external functions have no Python defaults. Supply every
        # argument even when the call has no keywords or omits trailing ones.
        result = []
        for param in self.params:
            if param in supplied:
                result.append(supplied[param])
            elif param in self.defaults:
                result.append(ast.Constant(self.defaults[param]))
            else:
                raise ValueError(
                    f"{label} is missing required {call} argument '{param}'")
        return result


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
