"""Single import boundary for Numba's internal compiler APIs."""

import numba.extending as extending
import numba.core.typing.templates as templates

from numba.core import cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.datamodel import models
from numba.core.imputils import lower_builtin, lower_constant
from numba.core.registry import cpu_target
from numba.core.typing.templates import (
    AbstractTemplate, AttributeTemplate,
    make_overload_method_template, signature,
)

# Numba exports these decorators dynamically, so its static declarations omit them.
infer_getattr = getattr(extending, "infer_getattr")
infer_global = getattr(templates, "infer_global")
lower_getattr = getattr(extending, "lower_getattr")
register_model = getattr(extending, "register_model")
typeof_impl = getattr(extending, "typeof_impl")

__all__ = [
    "AbstractTemplate", "AttributeTemplate", "cgutils", "cpu_target",
    "global_compiler_lock",
    "infer_getattr", "infer_global", "lower_builtin", "lower_constant",
    "lower_getattr", "register_model", "typeof_impl",
    "make_overload_method_template", "models", "signature",
]
