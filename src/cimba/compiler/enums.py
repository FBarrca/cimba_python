"""``enum.IntEnum`` members in compiled model code.

Numba types an ``IntEnum`` member as ``IntEnumMember``, represented by its
integer value, and already compiles comparisons, arithmetic and ``match``.
This module fills the gaps model code meets: ``int()``, ``float()`` and
``bool()``, casts to float and boolean (an enum passed where a float is
expected), and indexing arrays by a member. Fields of an ``IntEnum`` type are
typed as members by the record views (see ``views``).

Plain ``enum.Enum`` members keep Numba's own support (equality and
``.value``); converting one to a number names ``IntEnum`` as the fix.
"""

from __future__ import annotations

import operator

from numba import typeof, types
from numba.extending import overload

from .numba_compat import lower_cast


def member_type(kind) -> types.IntEnumMember:
    """The Numba type of ``kind``'s members, as Numba types the constants."""
    enum_class = typeof(kind)
    assert isinstance(enum_class, types.IntEnumClass)
    return enum_class.member_type


@lower_cast(types.IntEnumMember, types.Float)
def _enum_to_float(context, builder, fromty, toty, value):
    return context.cast(builder, value, fromty.dtype, toty)


@lower_cast(types.IntEnumMember, types.Boolean)
def _enum_to_boolean(context, builder, fromty, toty, value):
    return context.cast(builder, value, fromty.dtype, toty)


@lower_cast(types.Number, types.IntEnumMember)
@lower_cast(types.IntEnumMember, types.IntEnumMember)   # a different enum's member
def _number_to_enum(context, builder, fromty, toty, value):
    # Numba types record writes without checking convertibility; name the fix
    # instead of failing with a bare missing cast.
    enum_class = toty.instance_class.__name__
    raise TypeError(f"cannot store {fromty} as {enum_class}: assign a {enum_class} member")


def _plain_enum(value) -> bool:
    return (isinstance(value, types.EnumMember) and
            not isinstance(value, types.IntEnumMember))


def _conversion(convert, name: str):
    def typer(value):
        if isinstance(value, types.IntEnumMember):
            return lambda value: convert(value.value)
        if _plain_enum(value):
            enum_class = value.instance_class.__name__
            raise TypeError(
                f"{name}() of {enum_class} member: {enum_class} is not an "
                f"enum.IntEnum; derive it from enum.IntEnum to use its members "
                "as numbers in compiled model code")
        return None
    return typer


overload(int)(_conversion(int, "int"))
overload(float)(_conversion(float, "float"))
overload(bool)(_conversion(bool, "bool"))


@overload(operator.getitem)
def _getitem_by_member(array, index):
    if isinstance(array, types.Array) and isinstance(index, types.IntEnumMember):
        return lambda array, index: array[index.value]
    return None


@overload(operator.setitem)
def _setitem_by_member(array, index, value):
    if isinstance(array, types.Array) and isinstance(index, types.IntEnumMember):
        def implementation(array, index, value):
            array[index.value] = value
        return implementation
    return None


__all__ = ["member_type"]
