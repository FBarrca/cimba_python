"""Per-process data fields: ``sim.Struct``."""

from typing import TYPE_CHECKING, Any, Self

import numpy as np
from numba import carray, from_dtype, types
from numba.extending import overload as _nb_overload

from ._declarations import Handle, class_type_hints
from ._intrinsics import ptr_caster
from ._runtime import _PROC_DATA_OFFSET


class Struct:
    """Per-process data fields, declared like a dataclass: subclass it
    and annotate the fields (``float`` or ``int``). A process function
    asks for its own view by annotating a final parameter with the
    subclass::

        class Visitor(sim.Struct):
            patience: float
            rides: int

        class Park(sim.Model):
            @sim.process(copies=4)
            def visitor(self, vip: Visitor):
                vip.patience = cimba.random.triangular(0.5, 1.0, 1.5)

    Each process copy then carries its own fields, zeroed at creation, in
    the same native allocation as the process (this is the Python form of
    the C tutorial's ``struct visitor { struct cmb_process core; ... }``).
    Subclassing a Struct subclass inherits its fields.

    Other processes reach the same fields through the process handle:
    inside model code, ``Visitor(handle)`` returns a read/write view --
    so a handle pulled from a queue is all a server needs to update a
    visitor's statistics. ``@sim.process(struct=Visitor)`` attaches the
    fields without the view parameter.
    """

    if TYPE_CHECKING:
        def __init__(self, process: Handle) -> None: ...

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        raise TypeError(f"{cls.__name__}(handle) views are only available "
                        "inside compiled model code")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        fields = []
        for fname, hint in class_type_hints(cls).items():
            if fname.startswith("_"):
                continue
            if hint is float:
                fields.append((fname, "<f8"))
            elif hint is int:
                fields.append((fname, "<i8"))
            else:
                raise TypeError(f"struct field '{fname}': only float and "
                                "int fields are supported")
        if not fields:
            raise ValueError(f"struct '{cls.__name__}' declares no fields")
        cls._dtype = np.dtype(fields)
        cls._alloc_size = _PROC_DATA_OFFSET + cls._dtype.itemsize

        cast = ptr_caster(from_dtype(cls._dtype))
        offset = _PROC_DATA_OFFSET

        def view(process):
            return carray(cast(process + offset), 1)[0]

        @_nb_overload(cls)
        def struct_view(process):
            if not isinstance(process, types.Integer):
                return None
            return view


def _is_struct_class(obj: Any) -> bool:
    return (isinstance(obj, type) and issubclass(obj, Struct)
            and obj is not Struct)
