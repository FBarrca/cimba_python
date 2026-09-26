"""AST lowering of callbacks onto the flattened trial record.

Model and component callbacks are written against authoring-time paths
(``self.queue`` inside a component, ``self.zones[i].gates[j].queue`` in a
model callback). Before Numba compilation each callback is parsed, rewritten
against the flat record built by ``_components``, and re-exec'd:

* component ``@sim.process`` / ``@sim.collect`` / ``@sim.predicate`` /
  ``@sim.event`` methods become plain functions over the flattened env --
  ``self.queue`` becomes ``env.inlet__queue``. A collection's method
  compiles once (not once per item): ``self.queue`` lowers to
  ``env.stations__queue[__cimba_inst]``, where the instance index is
  recovered at runtime from the copy index (see ``_shared_instance_setup``);
* model callbacks that use component paths are rewritten the same way, with
  generated numpy tables (``_lowering_namespace``) backing dynamic item
  indices, per-item constants, and Ref/Refs dereferences;
* calls of ``@sim.function`` helpers keep their component syntax in user
  code and lower to explicit helper calls (see ``_functions``);
* method calls on entity and dataset fields lower to native helper calls
  (see ``_entity_methods``).

The rewritten source is kept in ``__cimba_source__`` and ``linecache`` so
tracebacks, Numba, and process-graph inference all read the lowered code.
"""

import ast
import copy
import inspect
import linecache
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, get_type_hints

import numpy as np
from numba import carray, njit

from ._callbacks import _callback_arg_count, _process_signature
from ._components import _ComponentRefDecl, _OwnerDecl, _offsets_from_counts
from ._entity_methods import METHOD_NAMES, helper_namespace, lower_method_calls

# --- Symbols shared with lowered code ----------------------------------------
#
# Lowered functions look up per-instance values that cannot be resolved
# to constants (dynamic item indices) in module-level numpy arrays
# published under these names by _lowering_namespace().

def _const_symbol(component: str, name: str) -> str:
    return f"_CIMBA_CONST_{component}__{name}"


def _pqueue_offsets_symbol(component: str, field_name: str) -> str:
    return f"_CIMBA_PQOFF_{component}__{field_name}"


def _process_offsets_symbol(component: str, field_name: str) -> str:
    return f"_CIMBA_PROCOFF_{component}__{field_name}"


def _collection_offsets_symbol(component: str) -> str:
    return f"_CIMBA_OFF_{component}"


def _collection_lengths_symbol(component: str) -> str:
    return f"_CIMBA_LEN_{component}"


def _component_slots_symbol(component: str) -> str:
    return f"_CIMBA_COMPSLOT_{component}"


def _field_slots_symbol(component: str, field_name: str) -> str:
    return f"_CIMBA_FIELDSLOT_{component}__{field_name}"


def _constant_slots_symbol(component: str, field_name: str) -> str:
    return f"_CIMBA_CONSTSLOT_{component}__{field_name}"


def _variant_slots_symbol(component: str) -> str:
    return f"_CIMBA_VARIANT_{component}"


def _ref_index_symbol(component: str, name: str) -> str:
    return f"_CIMBA_REFIDX_{component}__{name}"


def _ref_table_symbol(component: str, name: str) -> str:
    return f"_CIMBA_REFTAB_{component}__{name}"


def _ref_offsets_symbol(component: str, name: str) -> str:
    return f"_CIMBA_REFOFF_{component}__{name}"


def _ref_lengths_symbol(component: str, name: str) -> str:
    return f"_CIMBA_REFLEN_{component}__{name}"


def _lowering_namespace(components: Iterable[_OwnerDecl]) -> dict[str, Any]:
    """The numpy lookup tables a lowered function may reference, for the
    given decls, their descendants, and every decl reachable through
    Ref/Refs fields (whose symbols must be present too)."""
    namespace: dict[str, Any] = {}
    seen: set[int] = set()
    stack = list(components)
    while stack:
        root = stack.pop()
        for decl in root.walk():
            if id(decl) in seen:
                continue
            seen.add(id(decl))
            for name, values in decl.constants.items():
                if len(values) > 1:
                    namespace[_const_symbol(decl.name, name)] = \
                        np.asarray(values)
                slots = decl.constant_slots.get(name, ())
                if slots and slots != tuple(range(len(slots))):
                    namespace[_constant_slots_symbol(decl.name, name)] = \
                        np.asarray(slots, dtype=np.int64)
            for fname, slots in decl.field_slots.items():
                if slots != tuple(range(len(slots))):
                    namespace[_field_slots_symbol(decl.name, fname)] = \
                        np.asarray(slots, dtype=np.int64)
            for fname, offsets in decl.pqueue_offsets.items():
                if len(offsets) > 1:
                    namespace[_pqueue_offsets_symbol(decl.name, fname)] = \
                        np.asarray(offsets, dtype=np.int64)
            for fname, offsets in decl.process_offsets.items():
                if len(offsets) > 1:
                    namespace[_process_offsets_symbol(decl.name, fname)] = \
                        np.asarray(offsets, dtype=np.int64)
            if decl.collection and len(decl.parent_offsets) > 1:
                namespace[_collection_offsets_symbol(decl.name)] = np.asarray(
                    decl.parent_offsets, dtype=np.int64)
            if decl.collection and len(decl.parent_lengths) > 1:
                namespace[_collection_lengths_symbol(decl.name)] = np.asarray(
                    decl.parent_lengths, dtype=np.int64)
            if decl.parent_slots:
                namespace[_component_slots_symbol(decl.name)] = np.asarray(
                    decl.parent_slots, dtype=np.int64)
            if decl.polymorphic:
                namespace[_variant_slots_symbol(decl.name)] = np.asarray(
                    decl.specialization_slots(), dtype=np.int64)
            for name, ref in decl.component_refs.items():
                if ref.table:
                    if ref.table_decl is not None:
                        namespace[_ref_table_symbol(decl.name, name)] = \
                            np.asarray(ref.table_indices, dtype=np.int64)
                        stack.append(ref.table_decl)
                    if len(ref.table_offsets) > 1:
                        namespace[_ref_offsets_symbol(decl.name, name)] = \
                            np.asarray(ref.table_offsets, dtype=np.int64)
                    if len(ref.table_lengths) > 1:
                        namespace[_ref_lengths_symbol(decl.name, name)] = \
                            np.asarray(ref.table_lengths, dtype=np.int64)
                    continue
                targets = [t for t in ref.targets if t is not None]
                stack.extend(target_decl for target_decl, _index in targets)
                if len(ref.targets) > 1 and len(targets) == len(ref.targets):
                    first = targets[0][0]
                    if (all(t[0] is first for t in targets)
                            and first.count > 1):
                        namespace[_ref_index_symbol(decl.name, name)] = \
                            np.asarray([t[1] for t in targets],
                                       dtype=np.int64)
    return namespace


# --- AST lowering -------------------------------------------------------------
#
# _ComponentPathLowerer resolves component paths in an expression tree:
# a *namespace* (one component instance), a *collection* (must be
# indexed), or a Refs *table* (must be indexed), ending in a field or
# constant access that lowers to the flattened env field. Subclasses
# define the path roots: `self` inside component methods and model callbacks.

def _env_attr(env_name: str, field_name: str,
              ctx: ast.expr_context) -> ast.Attribute:
    return ast.Attribute(
        value=ast.Name(id=env_name, ctx=ast.Load()),
        attr=field_name,
        ctx=ctx,
    )


def _subscript(
    value: ast.Name | ast.Attribute | ast.Subscript,
    index: ast.expr,
    ctx: ast.expr_context,
) -> ast.Subscript:
    value.ctx = ast.Load()
    return ast.Subscript(value=value, slice=index, ctx=ctx)


def _literal_int(node: ast.AST | None) -> int | None:
    """The value of an ``int`` literal expression, else None."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


def _add(left: ast.expr, right: ast.expr) -> ast.expr:
    a, b = _literal_int(left), _literal_int(right)
    if a is not None and b is not None:
        return ast.Constant(a + b)
    return ast.BinOp(left=left, op=ast.Add(), right=right)


@dataclass(frozen=True)
class _OwnerAccess:
    """A resolved component-instance path: the decl plus the instance
    index expression (None when the decl has a single instance)."""

    decl: _OwnerDecl
    index: ast.expr | None
    text: str
    #: logical indexes this access may select when ``index`` is dynamic.
    possible_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _FieldAccess:
    """A resolved path to a declared field or captured constant."""

    decl: _OwnerDecl
    index: ast.expr | None
    field: str
    text: str
    possible_indices: tuple[int, ...] | None = None


@dataclass
class _FunctionSpec:
    """A Model/Component synchronous function lowered to one helper."""

    decl: _OwnerDecl
    name: str
    method: Callable[..., Any]
    graph_name: str
    symbol: str
    parameter_names: tuple[str, ...]
    argument_types: tuple[Any, ...]
    return_type: Any
    reads: tuple[_FieldAccess, ...]
    helper: Any
    callees: tuple[str, ...]
    #: None for the homogeneous shared helper; otherwise the recursive
    #: specialization ordinal and its logical instance indexes.
    variant: int | None = None
    instance_indices: tuple[int, ...] = ()
    dispatchers: dict[tuple[str, ...], tuple[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class _RefTableAccess:
    """A resolved path to a Refs table, before indexing."""

    parent: _OwnerAccess
    name: str
    ref: _ComponentRefDecl
    text: str


class _OwnerPathLowerer(ast.NodeTransformer):
    """Resolve component paths below a root (``_root_namespace_ref``,
    defined by subclasses) and lower them to flattened record accesses."""

    #: When set (method lowering with a runtime instance index), literal
    #: indices into a Refs table are checked against every possible instance.
    strict_ref_tables = False

    def __init__(
        self,
        *,
        env_name: str,
        label: str,
        functions: Mapping[str, _FunctionSpec],
    ):
        self.env_name = env_name
        #: the callback being lowered, for error messages
        self.label = label
        self.functions = functions
        self.called_functions: set[str] = set()
        self._ref_loop_tables: list[tuple[str, str, str | None, str]] = []

    def _root_namespace_ref(self, node: ast.AST) -> _OwnerAccess | None:
        raise NotImplementedError

    @staticmethod
    def _possible_positions(access: _OwnerAccess) -> tuple[int, ...]:
        if (position := _literal_int(access.index)) is not None:
            return (position,)
        if access.possible_indices is not None:
            return access.possible_indices
        return tuple(range(access.decl.count))

    @staticmethod
    def _ref_table_key(table: _RefTableAccess) -> tuple[str, str, str | None]:
        index = table.parent.index
        return (
            table.parent.decl.name,
            table.name,
            None if index is None else ast.dump(index),
        )

    def _lower_len_call(self, node: ast.Call) -> ast.expr | None:
        """Lower ``len`` for a resolved component collection or Refs table."""
        if (not isinstance(node.func, ast.Name)
                or node.func.id != "len"
                or len(node.args) != 1
                or node.keywords):
            return None
        collection = self._collection_ref(node.args[0])
        if collection is not None:
            return ast.copy_location(
                self._instance_table_expr(
                    collection.decl.parent_lengths,
                    collection.index,
                    _collection_lengths_symbol(collection.decl.name),
                    "component collection",
                ),
                node,
            )
        table = self._ref_table_ref(node.args[0])
        if table is not None:
            return ast.copy_location(
                self._instance_table_expr(
                    table.ref.table_lengths,
                    table.parent.index,
                    _ref_lengths_symbol(table.parent.decl.name,
                                        table.name),
                    "Refs table",
                ),
                node,
            )
        return None

    # -- path resolution -------------------------------------------------------

    def _present_position(
        self,
        parent: _OwnerAccess,
        values: Sequence[int],
        text: str,
        absent: Callable[[int], bool],
    ) -> int | None:
        """Validate a polymorphic child and return a static parent slot."""
        position = 0 if parent.index is None else _literal_int(parent.index)
        if position is not None:
            if absent(values[position]):
                raise ValueError(
                    f"{self.label} accesses {text}, which is not declared by that concrete component type"
                )
        elif any(absent(values[item]) for item in self._possible_positions(parent)):
            raise ValueError(
                f"{self.label} dynamically accesses {text}, which is not declared by every concrete component type"
            )
        return position

    def _namespace_ref(self, node: ast.AST) -> _OwnerAccess | None:
        root = self._root_namespace_ref(node)
        if root is not None:
            return root

        if isinstance(node, ast.Subscript):
            collection = self._collection_ref(node.value)
            if collection is not None:
                index = self._collection_item_index(
                    collection.decl, collection.index, node.slice
                )
                possible = None
                if _literal_int(index) is None:
                    if collection.index is None:
                        possible = tuple(range(collection.decl.count))
                    else:
                        possible_items: list[int] = []
                        for parent_index in self._possible_positions(
                                collection):
                            start = collection.decl.parent_offsets[parent_index]
                            length = collection.decl.parent_lengths[parent_index]
                            possible_items.extend(
                                range(start, start + length))
                        possible = tuple(possible_items)
                return _OwnerAccess(
                    collection.decl, index, f"{collection.text}[...]", possible
                )
            table = self._ref_table_ref(node.value)
            if table is not None:
                return self._ref_table_item(table, node.slice)
            return None

        if isinstance(node, ast.Attribute):
            parent = self._namespace_ref(node.value)
            if parent is None:
                return None
            child = parent.decl.child(node.attr)
            if child is not None:
                if child.collection:
                    return None
                text = f"{parent.text}.{node.attr}"
                if not child.parent_slots:
                    index = parent.index if child.count > 1 else None
                else:
                    position = self._present_position(
                        parent, child.parent_slots, text, lambda slot: slot < 0
                    )
                    if position is None:
                        # Only a dynamic (never a missing) parent index
                        # leaves the position unknown.
                        assert parent.index is not None
                        index = _subscript(
                            ast.Name(id=_component_slots_symbol(child.name),
                                     ctx=ast.Load()),
                            parent.index,
                            ast.Load(),
                        )
                    elif child.count > 1:
                        index = ast.Constant(child.parent_slots[position])
                    else:
                        index = None
                possible = None
                if _literal_int(index) is None:
                    possible = tuple(
                        child.parent_slots[position]
                        for position in self._possible_positions(parent)
                    )
                return _OwnerAccess(child, index, text, possible)
            ref = parent.decl.component_refs.get(node.attr)
            if ref is not None and not ref.table:
                return self._ref_namespace(parent, node.attr, ref)
            return None

        return None

    def _collection_ref(self, node: ast.AST) -> _OwnerAccess | None:
        if isinstance(node, ast.Attribute):
            parent = self._namespace_ref(node.value)
            if parent is None:
                return None
            child = parent.decl.child(node.attr)
            if child is None or not child.collection:
                return None
            if child.parent_lengths:
                self._present_position(
                    parent,
                    child.parent_lengths,
                    f"{parent.text}.{node.attr}",
                    lambda length: length == 0,
                )
            return _OwnerAccess(
                child,
                parent.index,
                f"{parent.text}.{node.attr}",
                parent.possible_indices,
            )

        return None

    def _ref_table_ref(self, node: ast.AST) -> _RefTableAccess | None:
        if not isinstance(node, ast.Attribute):
            return None
        parent = self._namespace_ref(node.value)
        if parent is None:
            return None
        ref = parent.decl.component_refs.get(node.attr)
        if ref is None or not ref.table:
            return None
        return _RefTableAccess(parent, node.attr, ref,
                               f"{parent.text}.{node.attr}")

    def _ref_namespace(
        self, parent: _OwnerAccess, name: str, ref: _ComponentRefDecl
    ) -> _OwnerAccess:
        """Dereference a Ref field: a static target when the instance is
        known, else an index lookup through the REFIDX table."""
        text = f"{parent.text}.{name}"
        index = parent.index
        position = 0 if index is None else _literal_int(index)
        if position is not None:
            target = ref.targets[position]
            if target is None:
                raise ValueError(
                    f"{self.label} dereferences {text}, which has no target for this instance"
                )
            target_decl, target_index = target
            target_expr = None if target_index is None else ast.Constant(target_index)
            return _OwnerAccess(target_decl, target_expr, text)
        assert index is not None  # a missing index is position 0
        parent_possible = self._possible_positions(parent)
        if any(ref.targets[position] is None
               for position in parent_possible):
            raise ValueError(
                f"{self.label} dereferences {text} with a dynamic instance index, but some instances have no target"
            )
        targets = [ref.targets[position] for position in parent_possible]
        first = targets[0][0]
        if any(target[0] is not first for target in targets):
            raise ValueError(
                f"{self.label} dereferences {text} with a "
                "dynamic instance index, which requires every instance to "
                "reference the same component declaration"
            )
        if first.count <= 1:
            return _OwnerAccess(first, None, text, (0,))
        lookup = _subscript(
            ast.Name(id=_ref_index_symbol(parent.decl.name, name), ctx=ast.Load()),
            index,
            ast.Load(),
        )
        return _OwnerAccess(first, lookup, text, tuple(target[1] for target in targets))

    def _ref_table_item(
        self, table: _RefTableAccess, item_slice: ast.expr
    ) -> _OwnerAccess:
        """Index a Refs table: a static target when both the instance and
        the entry are known, else a lookup through the REFTAB table."""
        item_index = self.visit(copy.deepcopy(item_slice))
        if not isinstance(item_index, ast.expr):
            raise TypeError("component refs table index did not lower to an expression")
        ref = table.ref
        text = f"{table.text}[...]"
        parent_index = table.parent.index
        parent_pos = 0 if parent_index is None else _literal_int(parent_index)
        item_pos = _literal_int(item_index)

        if parent_pos is not None and item_pos is not None:
            length = ref.table_lengths[parent_pos]
            position = item_pos
            if not 0 <= position < length:
                raise ValueError(
                    f"{self.label} index {position} is out of range for {table.text} (length {length})"
                )
            target_index = ref.table_indices[ref.table_offsets[parent_pos] + position]
            return _OwnerAccess(ref.table_decl, ast.Constant(target_index), text)

        if ref.table_decl is None:
            raise ValueError(
                f"{self.label} indexes {table.text}, which has no entries"
            )
        if parent_pos is None and self.strict_ref_tables:
            lengths = tuple(
                ref.table_lengths[position]
                for position in self._possible_positions(table.parent)
            )
            loop_bound = (
                isinstance(item_index, ast.Name)
                and (*self._ref_table_key(table), item_index.id)
                in self._ref_loop_tables
            )
            if len(set(lengths)) > 1 and not loop_bound:
                raise ValueError(
                    f"{self.label} indexes {table.text}, whose per-instance lengths differ"
                )
            if (
                item_pos is not None
                and any(
                    not 0 <= item_pos < ref.table_lengths[position]
                    for position in self._possible_positions(table.parent)
                )
            ):
                raise ValueError(
                    f"{self.label} index {item_pos} is out of range for {table.text} (lengths {lengths})"
                )
        if parent_pos is not None:
            offset: ast.expr = ast.Constant(ref.table_offsets[parent_pos])
        else:
            assert parent_index is not None  # a missing index is position 0
            offset = _subscript(
                ast.Name(
                    id=_ref_offsets_symbol(table.parent.decl.name, table.name),
                    ctx=ast.Load(),
                ),
                parent_index,
                ast.Load(),
            )
        lookup = _subscript(
            ast.Name(
                id=_ref_table_symbol(table.parent.decl.name, table.name), ctx=ast.Load()
            ),
            _add(offset, item_index),
            ast.Load(),
        )
        return _OwnerAccess(
            ref.table_decl, lookup, text, tuple(dict.fromkeys(ref.table_indices))
        )

    def _field_ref(self, node: ast.AST) -> _FieldAccess | None:
        if not isinstance(node, ast.Attribute):
            return None
        namespace = self._namespace_ref(node.value)
        if namespace is None:
            return None
        field_name = node.attr
        if (
            field_name in namespace.decl.direct_field_map
            or field_name in namespace.decl.constants
        ):
            return _FieldAccess(
                namespace.decl,
                namespace.index,
                field_name,
                f"{namespace.text}.{field_name}",
                namespace.possible_indices,
            )
        if namespace.decl.child(field_name) is not None:
            return None
        if field_name in namespace.decl.component_refs:
            return None
        self._raise_unknown_field(namespace, field_name)

    def _collection_item_index(
        self, decl: _OwnerDecl, parent_index: ast.expr | None, item_index: ast.expr
    ) -> ast.expr:
        """The flattened instance index of a collection item: the item
        index plus the parent instance's start offset."""
        index = self.visit(copy.deepcopy(item_index))
        if not isinstance(index, ast.expr):
            raise TypeError("component collection index did not lower to an expression")
        if (item := _literal_int(index)) is not None:
            length: int | None = None
            if len(decl.parent_lengths) <= 1:
                length = decl.parent_lengths[0] if decl.parent_lengths else 0
            elif (parent := _literal_int(parent_index)) is not None:
                length = decl.parent_lengths[parent]
            if length is not None and not 0 <= item < length:
                raise ValueError(
                    f"{self.label} collection index {item} is out of range (length {length})"
                )
        if len(decl.parent_offsets) <= 1:
            offset_value = decl.parent_offsets[0] if decl.parent_offsets else 0
            if offset_value == 0:
                return index
            return _add(ast.Constant(offset_value), index)
        if parent_index is None:
            raise TypeError("nested component collection has no parent index")
        if (parent := _literal_int(parent_index)) is not None:
            offset_value = decl.parent_offsets[parent]
            if offset_value == 0:
                return index
            return _add(ast.Constant(offset_value), index)
        offset = _subscript(
            ast.Name(id=_collection_offsets_symbol(decl.name),
                     ctx=ast.Load()),
            parent_index,
            ast.Load(),
        )
        return _add(offset, index)

    # -- lowered expressions ---------------------------------------------------

    def _instance_table_expr(
        self, values: Sequence[Any], index: ast.expr | None, symbol: str, what: str
    ) -> ast.expr:
        """A per-instance value: a constant when the instance is known,
        else an element of the numpy array published under `symbol`."""
        if len(values) == 1:
            return ast.Constant(values[0])
        if (position := _literal_int(index)) is not None:
            return ast.Constant(values[position])
        if index is None:
            raise TypeError(f"component {what} has no instance index")
        return _subscript(ast.Name(id=symbol, ctx=ast.Load()), index,
                          ast.Load())

    def _field_target(self, access: _FieldAccess, ctx: ast.expr_context) -> ast.expr:
        flat_name = access.decl.direct_field_map[access.field]
        target = _env_attr(self.env_name, flat_name, ctx)
        owners = access.decl.field_owners[access.field]
        if len(owners) <= 1:
            self._owned_index(access, owners)
            return target
        if access.index is None:
            raise TypeError("component field has no instance index")
        packed_index = self._owned_index(
            access,
            owners,
            access.decl.field_slots[access.field],
            _field_slots_symbol(access.decl.name, access.field),
        )
        assert packed_index is not None
        return _subscript(target, packed_index, ctx)

    def _owned_index(
        self,
        access: _FieldAccess,
        owners: tuple[int, ...],
        slots: tuple[int, ...] | None = None,
        slot_symbol: str | None = None,
    ) -> ast.expr | None:
        """Validate ownership and map a logical component index to storage."""
        index = access.index
        if (position := _literal_int(index)) is not None:
            if position not in owners:
                raise ValueError(
                    f"{self.label} accesses {access.text}, which is not declared by that concrete component type"
                )
            return ast.Constant(slots[position]) if slots is not None else index
        if index is not None:
            possible = set(access.possible_indices or range(access.decl.count))
            if not possible.issubset(owners):
                raise ValueError(
                    f"{self.label} dynamically accesses {access.text}, which is not declared by every concrete component type"
                )
        if index is None or slots is None or slots == tuple(range(len(slots))):
            return index
        assert slot_symbol is not None
        return _subscript(ast.Name(id=slot_symbol, ctx=ast.Load()), index, ast.Load())

    def _constant_expr(self, access: _FieldAccess) -> ast.expr:
        owners = access.decl.constant_owners[access.field]
        slots = access.decl.constant_slots[access.field]
        index = self._owned_index(
            access,
            owners,
            slots if len(owners) > 1 else None,
            _constant_slots_symbol(access.decl.name, access.field),
        )
        return self._instance_table_expr(
            access.decl.constants[access.field],
            index,
            _const_symbol(access.decl.name, access.field),
            "constant",
        )

    def _lower_indexed_field(
        self, access: _FieldAccess, node: ast.Subscript
    ) -> ast.Subscript | None:
        """Lower ``<pqueues/processes field>[i]`` to an element of the
        flattened shared array, at the instance's offset plus ``i``."""
        decl = access.decl
        kind = decl.decls.kind_of(access.field)
        if kind == "pqueues":
            what = "PQueues"
            offsets = decl.pqueue_offsets[access.field]
            symbol = _pqueue_offsets_symbol(decl.name, access.field)
        elif kind == "processes":
            what = "Processes"
            offsets = decl.process_offsets[access.field]
            symbol = _process_offsets_symbol(decl.name, access.field)
        else:
            return None
        item = self.visit(copy.deepcopy(node.slice))
        if not isinstance(item, ast.expr):
            raise TypeError(f"component {what} index did not lower to an expression")
        offset = self._instance_table_expr(
            offsets, access.index, symbol, f"{what} field"
        )
        if isinstance(offset, ast.Constant) and offset.value == 0:
            index = item
        else:
            index = _add(offset, item)
        flat = _env_attr(self.env_name,
                         decl.direct_field_map[access.field], ast.Load())
        return ast.copy_location(_subscript(flat, index, node.ctx), node)

    def _raise_unknown_field(self, namespace: _OwnerAccess, field_name: str) -> None:
        kind = (
            "component collection field"
            if namespace.decl.collection
            else "component field"
        )
        raise ValueError(
            f"{self.label} references unknown {kind} {namespace.text}.{field_name}"
        )

    def _function_specs(
        self, namespace: _OwnerAccess, method_name: str
    ) -> tuple[_FunctionSpec, ...]:
        candidates = [
            spec
            for spec in self.functions.values()
            if spec.decl is namespace.decl and spec.name == method_name
        ]
        if not candidates:
            return ()
        if (position := _literal_int(namespace.index)) is not None:
            return tuple(
                spec
                for spec in candidates
                if position in spec.instance_indices
            )
        shared = [spec for spec in candidates if spec.variant is None]
        if shared:
            return (shared[0],)
        possible = set(namespace.possible_indices
                       or range(namespace.decl.count))
        ordered = sorted(
            (
                spec
                for spec in candidates
                if possible.intersection(spec.instance_indices)
            ),
            # Shared specs (variant None) were returned above.
            key=lambda spec: spec.variant or 0,
        )
        covered = {
            index for spec in ordered for index in spec.instance_indices
            if index in possible
        }
        if covered != possible:
            raise ValueError(
                f"{self.label} dynamically calls "
                f"{namespace.text}.{method_name}(), which is not declared "
                "as @sim.function by every concrete component type"
            )
        return tuple(ordered)

    def _lower_one_component_function_call(
        self, node: ast.Call, receiver: _OwnerAccess, spec: _FunctionSpec
    ) -> ast.Call:
        self._bind_function(spec)
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=spec.symbol, ctx=ast.Load()),
                args=[
                    ast.Name(id=self.env_name, ctx=ast.Load()),
                    copy.deepcopy(receiver.index) if receiver.index is not None
                    else ast.Constant(0),
                    *(self.visit(copy.deepcopy(arg)) for arg in node.args),
                ],
                keywords=[],
            ),
            node,
        )

    def _bind_function(self, spec: _FunctionSpec) -> None:
        self.called_functions.add(spec.graph_name)

    def _validate_function_call(self, node: ast.Call, spec: _FunctionSpec) -> None:
        if node.keywords:
            raise ValueError(
                f"{self.label} call to component function '{spec.graph_name}' must use positional arguments"
            )
        if len(node.args) != len(spec.parameter_names):
            raise ValueError(
                f"{self.label} call to component function '{spec.graph_name}' takes {len(spec.parameter_names)} argument(s), got {len(node.args)}"
            )

    def _dispatch_function_call(
        self,
        node: ast.Call,
        receiver: _OwnerAccess,
        specs: Sequence[_FunctionSpec],
    ) -> ast.expr:
        self._validate_function_call(node, specs[0])
        if len(specs) == 1:
            return self._lower_one_component_function_call(node, receiver, specs[0])
        first = specs[0]
        contract = (first.parameter_names, first.argument_types,
                    first.return_type)
        if any((spec.parameter_names, spec.argument_types, spec.return_type)
               != contract for spec in specs[1:]):
            raise TypeError(
                f"{self.label} dynamically calls {receiver.text}.{first.name}(), whose concrete implementations have incompatible signatures"
            )
        if receiver.index is None:
            raise TypeError("polymorphic component function has no index")
        key = tuple(spec.graph_name for spec in specs)
        dispatch = first.dispatchers.get(key)
        if dispatch is None:
            symbol = f"{first.symbol}_dispatch_{len(first.dispatchers)}"
            parameters = ["__cimba_env", "__cimba_index", *first.parameter_names]
            arguments = ", ".join(parameters)
            table = _variant_slots_symbol(receiver.decl.name)
            lines = [f"def {symbol}({arguments}):"]
            for spec in specs[:-1]:
                lines.extend([
                    f"    if {table}[__cimba_index] == {spec.variant}:",
                    f"        return {spec.symbol}({arguments})",
                ])
            lines.append(f"    return {specs[-1].symbol}({arguments})")
            namespace = _lowering_namespace((receiver.decl,))
            namespace.update({spec.symbol: spec.helper for spec in specs})
            exec("\n".join(lines), namespace)
            dispatch = symbol, njit(namespace[symbol])
            first.dispatchers[key] = dispatch
        # Resolve callees for graph metadata and the helper-body namespace.
        for spec in specs:
            self._bind_function(spec)
        return ast.copy_location(ast.Call(
            func=ast.Name(id=dispatch[0], ctx=ast.Load()),
            args=[ast.Name(id=self.env_name, ctx=ast.Load()),
                  copy.deepcopy(receiver.index),
                  *(self.visit(copy.deepcopy(arg)) for arg in node.args)],
            keywords=[]), node)

    # -- node visitors -----------------------------------------------------------

    def visit_For(self, node: ast.For) -> ast.AST:
        """Track the canonical ``range(len(refs))`` loop bound.

        The runtime table offset makes dynamic indexing valid for the
        current owner even when different owners have different table
        lengths.  The marker is intentionally scoped to the loop body so
        an index variable used after the loop does not inherit that proof.
        """
        loop_ref: tuple[str, str, str | None, str] | None = None
        if (isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and len(node.iter.args) == 1
                and not node.iter.keywords
                and isinstance(node.iter.args[0], ast.Call)):
            length_call = node.iter.args[0]
            if (isinstance(length_call.func, ast.Name)
                    and length_call.func.id == "len"
                    and len(length_call.args) == 1
                    and not length_call.keywords):
                table = self._ref_table_ref(length_call.args[0])
                if table is not None:
                    decl_name, table_name, parent_key = \
                        self._ref_table_key(table)
                    loop_ref = (decl_name, table_name, parent_key,
                                node.target.id)

        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        if loop_ref is not None:
            self._ref_loop_tables.append(loop_ref)
        node.body = [self.visit(statement) for statement in node.body]
        if loop_ref is not None:
            self._ref_loop_tables.pop()
        node.orelse = [self.visit(statement) for statement in node.orelse]
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        lowered_len = self._lower_len_call(node)
        if lowered_len is not None:
            return lowered_len
        if isinstance(node.func, ast.Name) and node.func.id == "getattr" and node.args:
            target = (
                self._namespace_ref(node.args[0])
                or self._collection_ref(node.args[0])
                or self._ref_table_ref(node.args[0])
            )
            if target is not None:
                raise ValueError(
                    f"{self.label} uses dynamic getattr({target.text}, ...), which is not supported"
                )
        if isinstance(node.func, ast.Attribute):
            receiver = self._namespace_ref(node.func.value)
            if receiver is not None:
                specs = self._function_specs(receiver, node.func.attr)
                if specs:
                    return self._dispatch_function_call(node, receiver, specs)
            access = self._field_ref(node.func.value)
            if access is not None and access.decl.owner_root:
                node.args = [self.visit(arg) for arg in node.args]
                node.keywords = [
                    ast.keyword(arg=item.arg, value=self.visit(item.value))
                    for item in node.keywords
                ]
                return node
            if access is not None:
                kind = access.decl.decls.kind_of(access.field)
                if kind == "pqueues" or (
                    access.decl.decls.fields[access.field].kind.binding is None
                    and kind not in ("condition", "event")
                ):
                    raise ValueError(
                        f"{self.label} cannot call {access.text}.{node.func.attr}() inside compiled code"
                    )
                return ast.copy_location(
                    ast.Call(
                        func=ast.Attribute(
                            value=self._field_target(access, ast.Load()),
                            attr=node.func.attr,
                            ctx=ast.Load(),
                        ),
                        args=[self.visit(arg) for arg in node.args],
                        keywords=[
                            ast.keyword(arg=item.arg, value=self.visit(item.value))
                            for item in node.keywords
                        ],
                    ),
                    node,
                )
            target = (
                self._namespace_ref(node.func.value)
                or self._collection_ref(node.func.value)
                or self._ref_table_ref(node.func.value)
            )
            if target is not None:
                raise ValueError(
                    f"{self.label} cannot call {target.text}.{node.func.attr}() inside compiled code"
                )
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        access = self._field_ref(node.value)
        if access is not None:
            lowered = self._lower_indexed_field(access, node)
            if lowered is not None:
                return lowered
        collection = self._collection_ref(node.value)
        if collection is not None:
            raise ValueError(
                f"{self.label} uses {collection.text}[...] directly; access one of its fields"
            )
        table = self._ref_table_ref(node.value)
        if table is not None:
            raise ValueError(
                f"{self.label} uses {table.text}[...] directly; access one of its fields"
            )
        return self.generic_visit(node)

    def _lower_attribute(self, access: _FieldAccess, node: ast.Attribute) -> ast.AST:
        if access.decl.decls.kind_of(access.field) in ("pqueues", "processes"):
            raise ValueError(
                f"{self.label} must index {access.text} before using it"
            )
        if access.field in access.decl.constants:
            if not isinstance(node.ctx, ast.Load):
                raise ValueError(
                    f"{self.label} cannot assign to constant {access.text}"
                )
            return ast.copy_location(self._constant_expr(access), node)
        return ast.copy_location(self._field_target(access, node.ctx), node)

    def _direct_path_error(self, kind: str, text: str) -> ValueError:
        suffix = {
            "namespace": "directly; access one of its fields",
            "collection": "directly; index it and access one of its fields",
            "table": "before using it",
        }[kind]
        action = "must index" if kind == "table" else "cannot use"
        return ValueError(f"{self.label} {action} {text} {suffix}")

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        nested = self._field_ref(node.value)
        if nested is not None:
            raise ValueError(
                f"{self.label} cannot access attributes below component field {nested.text}"
            )
        access = self._field_ref(node)
        if access is not None:
            return self._lower_attribute(access, node)
        for kind, resolve in (
            ("namespace", self._namespace_ref),
            ("collection", self._collection_ref),
            ("table", self._ref_table_ref),
        ):
            if (path := resolve(node)) is not None:
                raise self._direct_path_error(kind, path.text)
        return self.generic_visit(node)


class _CallbackLowerer(_OwnerPathLowerer):
    """Lower one model or component callback. ``receiver_name`` (``self``)
    is the owner -- a component instance selected by ``instance_index``, or
    the model itself -- and ``env_name`` the trial record, whose paths start
    at the model."""

    def __init__(
        self,
        *,
        label: str,
        env_name: str,
        receiver_name: str,
        decl: _OwnerDecl,
        model_decl: _OwnerDecl,
        instance_index: ast.expr,
        possible_indices: tuple[int, ...],
        functions: Mapping[str, _FunctionSpec],
    ):
        super().__init__(env_name=env_name, label=label, functions=functions)
        self.receiver_name = receiver_name
        self.decl = decl
        self.model_decl = model_decl
        self.instance_index = instance_index
        self.possible_indices = possible_indices
        self.strict_ref_tables = not isinstance(instance_index, ast.Constant)
        self.changed = False

    def _root_namespace_ref(self, node: ast.AST) -> _OwnerAccess | None:
        if not isinstance(node, ast.Name):
            return None
        if node.id == self.receiver_name:
            index = (copy.deepcopy(self.instance_index)
                     if self.decl.count > 1 else None)
            return _OwnerAccess(self.decl, index, self.receiver_name,
                                self.possible_indices)
        if node.id == self.env_name:
            return _OwnerAccess(self.model_decl, None, self.env_name, (0,))
        return None

    def _raise_unknown_field(self, namespace: _OwnerAccess, field_name: str) -> None:
        if self.decl.owner_root and not namespace.decl.owner_root:
            return super()._raise_unknown_field(namespace, field_name)
        raise ValueError(
            f"{self.label} references unsupported {namespace.text}.{field_name}"
        )

    def visit(self, node: ast.AST) -> ast.AST:
        lowered = super().visit(node)
        if lowered is not node:
            self.changed = True
        return lowered

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not self.decl.owner_root and node.id == self.receiver_name:
            raise ValueError(
                f"{self.label} cannot use self directly inside compiled code"
            )
        return node


# --- Codegen ------------------------------------------------------------------
#
# The lowered ASTs are unparsed, exec'd, and returned as plain functions
# whose source (kept in __cimba_source__ and linecache) reflects the
# rewrite -- Numba and the process-DAG inference both re-read it.

def _function_calls(fn: Callable[..., Any]) -> tuple[str, ...]:
    """Graph names of the ``@sim.function`` helpers a lowered callback
    calls, recorded by lowering for process graphs and diagnostics."""
    return getattr(fn, "__cimba_function_calls__", ())


def _set_function_calls(fn: Callable[..., Any], calls: Iterable[str]) -> None:
    setattr(fn, "__cimba_function_calls__", tuple(sorted(set(calls))))


def _closure_namespace(fn: Callable[..., Any]) -> dict[str, Any]:
    namespace = dict(fn.__globals__)
    if fn.__closure__ is not None:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            namespace[name] = cell.cell_contents
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
    raise ValueError(f"callback '{fn.__qualname__}' source does not contain "
                     "a function definition")


def _component_method_source(fn: Callable[..., Any],
                             kind: str) -> ast.FunctionDef:
    try:
        return _function_def_from_source(fn)
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"component {kind} '{fn.__qualname__}' needs inspectable source"
        ) from exc


def _compile_lowered(
    node: ast.FunctionDef,
    *,
    filename: str,
    fn_name: str,
    qualname: str,
    namespace: dict[str, Any],
    like: Callable[..., Any],
) -> Callable[..., Any]:
    """Exec a lowered FunctionDef and return the generated function; the
    source goes into linecache so tracebacks and inspect.getsource()
    resolve against the rewritten code."""
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    source = ast.unparse(module) + "\n"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    exec(compile(source, filename, "exec"), namespace)
    generated = namespace[fn_name]
    generated.__module__ = like.__module__
    generated.__qualname__ = qualname
    generated.__cimba_source__ = source
    if calls := _function_calls(like):
        _set_function_calls(generated, calls)
    return generated


def _direct_process_callback(fn: Callable[..., Any], name: str,
                             struct_view: type | None) -> Callable[..., Any]:
    """Build a non-indexed process body with the native process ABI
    ``(me, ctxp)``: the record view and, when the process asks for one, its
    own ``sim.Struct`` view are bound before the inlined body runs."""
    fn_node = _function_def_from_source(fn)
    env_name = fn_node.args.args[0].arg
    body = copy.deepcopy(fn_node.body)
    prefix = ast.parse(f"{env_name} = carray(ctxp, 1)[0]").body
    namespace = _closure_namespace(fn)
    namespace["carray"] = carray
    if struct_view is not None:
        view_name = fn_node.args.args[-1].arg
        prefix.extend(ast.parse(f"{view_name} = _CIMBA_STRUCT_VIEW(me)").body)
        namespace["_CIMBA_STRUCT_VIEW"] = struct_view
    wrapper = ast.parse(f"def _cimba_direct_{name}(me, ctxp):\n    pass").body[0]
    assert isinstance(wrapper, ast.FunctionDef)
    wrapper.body = prefix + body + ast.parse("return 0").body
    return _compile_lowered(
        wrapper,
        filename=f"<cimba direct process '{name}'>",
        fn_name=wrapper.name,
        qualname=wrapper.name,
        namespace=namespace,
        like=fn,
    )


def _direct_collect_callback(fn: Callable[..., Any], index: int,
                             count: int) -> Callable[..., Any]:
    """Build a collect body with the native callback ABI.

    Multi-instance component collectors execute their lowered body in one
    native callback loop, avoiding a separate lazy ``njit`` dispatcher and
    repeated generated calls from trial teardown.
    """
    fn_node = _function_def_from_source(fn)
    env_name = fn_node.args.args[0].arg
    namespace = _closure_namespace(fn)
    namespace["carray"] = carray
    body = ast.parse(f"{env_name} = carray(ctxp, 1)[0]").body
    if count == 1:
        body.extend(copy.deepcopy(fn_node.body))
    else:
        if len(fn_node.args.args) < 2:
            raise TypeError(
                "multi-instance collect callback needs an index argument"
            )
        index_name = fn_node.args.args[1].arg
        loop_index = f"_cimba_collect_index_{index}"
        loop = ast.parse(
            f"for {loop_index} in range({count}):\n    {index_name} = {loop_index}"
        ).body[0]
        assert isinstance(loop, ast.For)
        loop.body.extend(copy.deepcopy(fn_node.body))
        body.append(loop)
    wrapper = ast.parse(f"def _cimba_direct_collect_{index}(ctxp):\n    pass").body[
        0
    ]
    assert isinstance(wrapper, ast.FunctionDef)
    wrapper.body = body
    return _compile_lowered(
        wrapper,
        filename=f"<cimba direct collect {index}>",
        fn_name=wrapper.name,
        qualname=wrapper.name,
        namespace=namespace,
        like=fn,
    )


def _strip_function_annotations(node: ast.FunctionDef) -> None:
    node.decorator_list = []
    node.returns = node.type_comment = None
    for arg in node.args.args:
        arg.annotation = arg.type_comment = None


@dataclass(frozen=True)
class _CallbackLoweringContext:
    #: entity and dataset field name -> kind, for method-call lowering
    method_fields: Mapping[str, str]
    owner_decl: _OwnerDecl
    functions: Mapping[str, _FunctionSpec]


def _lower_owner_methods(
    node: ast.FunctionDef,
    fn: Callable[..., Any],
    *,
    env_name: str,
    label: str,
    owner_name: str,
    context: _CallbackLoweringContext,
    namespace: dict[str, Any],
) -> tuple[ast.FunctionDef, bool]:
    """Lower method calls on entity/dataset fields in a callback and in the
    plain Numba helpers it references."""
    helpers_changed = _rewire_entity_method_helpers(
        namespace,
        set(fn.__code__.co_names),
        model_name=owner_name,
        method_fields=context.method_fields,
        cache={},
    )
    node, lowered = lower_method_calls(
        node, env_name=env_name, fields=context.method_fields, label=label)
    if lowered or helpers_changed:
        namespace.update(helper_namespace())
    return node, lowered or helpers_changed


def _lower_component_method(
    node: ast.FunctionDef,
    *,
    kind: str,
    component_name: str,
    component_decl: _OwnerDecl,
    instance_index: ast.expr,
    possible_indices: tuple[int, ...],
    method_name: str,
    method: Callable[..., Any],
    struct_view: type | None = None,
    prologue: Sequence[ast.stmt] = (),
    extra_namespace: Mapping[str, Any] | None = None,
    context: _CallbackLoweringContext,
) -> Callable[..., Any]:
    """Shared tail of process/collect lowering: drop `self`, rewrite the
    body against the flattened env, and compile the result."""
    args = node.args
    receiver_name = args.args[0].arg
    env_name = args.args[1].arg
    fn_name = f"{component_name}__{method_name}"
    node.name = fn_name
    node.decorator_list = []
    node.returns = None
    node.type_comment = None
    args.args = args.args[1:]
    for index, arg in enumerate(args.args):
        if struct_view is not None and index == len(args.args) - 1:
            # Keep an annotation on the view parameter so process registration
            # detects it on the lowered function; the exec namespace maps
            # _CIMBA_STRUCT_VIEW to the struct class.
            arg.annotation = ast.Name(id="_CIMBA_STRUCT_VIEW",
                                      ctx=ast.Load())
        else:
            arg.annotation = None
        arg.type_comment = None

    lowerer = _CallbackLowerer(
        label=f"component '{component_name}' {kind}",
        receiver_name=receiver_name,
        env_name=env_name,
        decl=component_decl,
        model_decl=context.owner_decl,
        instance_index=instance_index,
        possible_indices=possible_indices,
        functions=context.functions,
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError(f"component {kind} lowering produced a non-function")
    label = f"component {kind} '{component_name}.{method_name}'"
    namespace = _closure_namespace(method)
    lowered, _ = _lower_owner_methods(
        lowered,
        method,
        env_name=env_name,
        label=label,
        owner_name=component_name,
        context=context,
        namespace=namespace,
    )
    lowered.body[:0] = list(prologue)

    if struct_view is not None:
        namespace["_CIMBA_STRUCT_VIEW"] = struct_view
    if extra_namespace:
        namespace.update(extra_namespace)
    namespace.update(
        _model_lowering_namespace(
            {component_decl.name: component_decl}, context.functions
        )
    )
    generated = _compile_lowered(
        lowered,
        filename=f"<cimba component '{component_name}.{method_name}'>",
        fn_name=fn_name,
        qualname=fn_name,
        namespace=namespace,
        like=method,
    )
    _set_function_calls(generated, lowerer.called_functions)
    return generated


def _shared_instance_setup(
    node: ast.FunctionDef,
    base: str,
    counts: tuple[int, ...],
    base_arg_count: int,
    instance_indices: tuple[int, ...],
) -> tuple[ast.expr, list[ast.stmt], dict[str, Any]]:
    """Map global process-copy indexes to collection items/local copies."""
    params = node.args.args
    user_idx = params[2].arg if base_arg_count == 3 else None
    if user_idx is not None:
        params[2] = ast.arg(arg="__cimba_idx")
    else:
        params.insert(2, ast.arg(arg="__cimba_idx"))

    inst_symbol = f"_CIMBA_PROCINST_{base}"
    copybase_symbol = f"_CIMBA_COPYBASE_{base}"
    group_symbol = f"_CIMBA_PROCGROUP_{base}"
    mapped = instance_indices != tuple(range(len(instance_indices)))
    local_inst = "__cimba_local_inst" if mapped else "__cimba_inst"
    uniform = len(set(counts)) == 1
    per_instance = counts[0]
    lines: list[str] = []
    tables: dict[str, Any] = {}

    # __cimba_inst: the collection item this global copy belongs to.
    if uniform and per_instance == 1:
        lines.append(f"{local_inst} = __cimba_idx")
    elif uniform:
        lines.append(f"{local_inst} = __cimba_idx // {per_instance}")
    else:
        tables[inst_symbol] = np.repeat(
            np.arange(len(counts), dtype=np.int64),
            np.asarray(counts, dtype=np.int64))
        lines.append(f"{local_inst} = {inst_symbol}[__cimba_idx]")

    # the user's copy index: this copy's position within its own item.
    if user_idx is not None:
        if uniform and per_instance == 1:
            lines.append(f"{user_idx} = 0")
        elif uniform:
            lines.append(f"{user_idx} = __cimba_idx % {per_instance}")
        else:
            tables[copybase_symbol] = np.asarray(
                _offsets_from_counts(counts)[1], dtype=np.int64)
            lines.append(
                f"{user_idx} = __cimba_idx - "
                f"{copybase_symbol}[{local_inst}]")

    if mapped:
        tables[group_symbol] = np.asarray(
            instance_indices, dtype=np.int64)
        lines.append(f"__cimba_inst = {group_symbol}[{local_inst}]")

    return (ast.Name(id="__cimba_inst", ctx=ast.Load()),
            ast.parse("\n".join(lines)).body, tables)


def _component_process_signature(
    component_name: str,
    method_name: str,
    method: Callable[..., Any],
) -> tuple[type | None, int]:
    """Validate a component process method's ``(self, env[, idx][, view])``
    signature and return ``(struct_view_class_or_None, base_arg_count)``."""
    signature = (f"component process '{component_name}.{method_name}' must "
                 "take (self, env), (self, env, idx), and optionally a "
                 "final sim.Struct view parameter, without defaults")
    return _process_signature(
        method, 2,
        f"component process '{component_name}.{method_name}'", signature)


def _lower_component_process(
    component_name: str,
    component_decl: _OwnerDecl,
    method_name: str,
    method: Callable[..., Any],
    *,
    instance_index: int | None = None,
    copies_per_instance: tuple[int, ...] | None = None,
    instance_indices: tuple[int, ...] | None = None,
    context: _CallbackLoweringContext,
) -> Callable[..., Any]:
    """Lower a component process specialized to ``instance_index``, or
    shared by the ``instance_indices`` group with ``copies_per_instance``
    copies each."""
    node = copy.deepcopy(_component_method_source(method, "process"))
    struct_view, base_arg_count = _component_process_signature(
        component_name, method_name, method)

    if instance_index is not None:
        index_expr: ast.expr = ast.Constant(instance_index)
        possible: tuple[int, ...] = (instance_index,)
        prologue: list[ast.stmt] = []
        tables: dict[str, Any] = {}
    else:
        assert instance_indices is not None and copies_per_instance is not None
        index_expr, prologue, tables = _shared_instance_setup(
            node, f"{component_name}__{method_name}",
            tuple(copies_per_instance), base_arg_count, instance_indices)
        possible = instance_indices

    return _lower_component_method(
        node, kind="process", component_name=component_name,
        component_decl=component_decl, instance_index=index_expr,
        possible_indices=possible,
        method_name=method_name, method=method, struct_view=struct_view,
        prologue=prologue, extra_namespace=tables,
        context=context)


def _lower_component_collect(
    component_name: str,
    component_decl: _OwnerDecl,
    method_name: str,
    method: Callable[..., Any],
    *,
    instance_index: int | None = None,
    instance_indices: tuple[int, ...] | None = None,
    context: _CallbackLoweringContext,
) -> Callable[..., Any]:
    """Lower a component collect method specialized to ``instance_index``,
    or shared by the ``instance_indices`` group -- one function that takes
    the position within the group as its second argument."""
    node = copy.deepcopy(_component_method_source(method, "collect"))
    args = node.args
    signature = (f"component collect '{component_name}.{method_name}' must "
                 "take (self, env) without defaults")
    _callback_arg_count(method, (2,), signature)
    if instance_index is not None:
        index_expr: ast.expr = ast.Constant(instance_index)
        possible: tuple[int, ...] = (instance_index,)
        prologue: list[ast.stmt] = []
        tables: dict[str, Any] = {}
    else:
        assert instance_indices is not None
        args.args.append(ast.arg(arg="__cimba_group_inst"))
        possible = indices = instance_indices
        if indices == tuple(range(len(indices))):
            index_expr = ast.Name(
                id="__cimba_group_inst", ctx=ast.Load())
            prologue = []
            tables = {}
        else:
            symbol = f"_CIMBA_COLLECTGROUP_{component_name}__{method_name}"
            index_expr = ast.Name(id="__cimba_inst", ctx=ast.Load())
            prologue = ast.parse(
                f"__cimba_inst = {symbol}[__cimba_group_inst]").body
            tables = {
                symbol: np.asarray(indices, dtype=np.int64),
            }
    return _lower_component_method(
        node, kind="collect", component_name=component_name,
        component_decl=component_decl, instance_index=index_expr,
        possible_indices=possible,
        method_name=method_name, method=method,
        prologue=prologue, extra_namespace=tables,
        context=context)


def _lower_component_signal(
    component_name: str,
    component_decl: _OwnerDecl,
    method_name: str,
    method: Callable[..., Any],
    *,
    kind: str,
    instance_index: int,
    context: _CallbackLoweringContext,
) -> Callable[..., Any]:
    """Lower one instance of a component predicate or event callback."""
    node = copy.deepcopy(_component_method_source(method, kind))
    arity = (2,) if kind == "predicate" else (2, 3)
    suffix = " or (self, env, data)" if kind == "event" else ""
    signature = f"component {kind} '{component_name}.{method_name}' must take (self, env){suffix} without defaults"
    _callback_arg_count(method, arity, signature)
    if kind == "predicate" and get_type_hints(method).get("return") is not bool:
        raise ValueError(
            f"component predicate '{component_name}.{method_name}' must return bool"
        )
    generated = _lower_component_method(
        node,
        kind=kind,
        component_name=component_name,
        component_decl=component_decl,
        instance_index=ast.Constant(instance_index),
        possible_indices=(instance_index,),
        method_name=method_name,
        method=method,
        context=context,
    )
    if kind == "predicate":
        generated.__annotations__["return"] = bool
    return generated


def _lower_model_component_refs_in_node(
    node: ast.FunctionDef,
    *,
    model_name: str,
    owner_decl: _OwnerDecl,
    functions: Mapping[str, _FunctionSpec],
) -> tuple[ast.FunctionDef, bool, tuple[str, ...]]:
    if not node.args.args:
        return node, False, ()

    env_name = node.args.args[0].arg
    lowerer = _CallbackLowerer(
        label=f"model '{model_name}' callback '{node.name}'",
        receiver_name=env_name,
        env_name=env_name,
        decl=owner_decl,
        model_decl=owner_decl,
        instance_index=ast.Constant(0),
        possible_indices=(0,),
        functions=functions,
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError("model callback lowering produced a non-function")
    return lowered, lowerer.changed, tuple(sorted(lowerer.called_functions))


def _model_lowering_namespace(
    component_roots: Mapping[str, _OwnerDecl],
    functions: Mapping[str, _FunctionSpec] | None,
) -> dict[str, Any]:
    namespace = helper_namespace()
    if functions:
        namespace.update({spec.symbol: spec.helper for spec in functions.values()})
        for spec in functions.values():
            namespace.update(spec.dispatchers.values())
    namespace.update(_lowering_namespace(component_roots.values()))
    return namespace


def _compile_model_callback(
    fn: Callable[..., Any],
    lowered: ast.FunctionDef,
    model_name: str,
    namespace: dict[str, Any],
) -> Callable[..., Any]:
    """Exec a lowered model callback in its (augmented) closure namespace."""
    _strip_function_annotations(lowered)
    return _compile_lowered(
        lowered,
        filename=f"<cimba model callback '{model_name}.{fn.__name__}'>",
        fn_name=fn.__name__,
        qualname=fn.__qualname__,
        namespace=namespace,
        like=fn,
    )


def _lower_entity_method_helper(
    helper: Any,
    *,
    model_name: str,
    method_fields: Mapping[str, str],
    cache: dict[int, Any],
) -> Any:
    """Recursively lower method sugar inside a referenced Numba helper."""
    py_func = getattr(helper, "py_func", None)
    if py_func is None:
        return helper
    key = id(py_func)
    if key in cache:
        return cache[key]
    # Guard recursive/mutually-recursive helpers: while we're rewriting
    # this one, references to it (including from itself) resolve to the
    # original -- an accepted limitation for the exotic self-recursive case.
    cache[key] = helper
    names = set(py_func.__code__.co_names)
    direct = bool(names.intersection(method_fields)
                 and names.intersection(METHOD_NAMES))
    namespace = _closure_namespace(py_func)
    helpers_changed = _rewire_entity_method_helpers(
        namespace, names, model_name=model_name, method_fields=method_fields,
        cache=cache)
    if not direct and not helpers_changed:
        return helper
    try:
        node = copy.deepcopy(_function_def_from_source(py_func))
    except (OSError, TypeError):
        return helper
    if not node.args.args:
        return helper

    env_name = node.args.args[0].arg
    lowered, changed = lower_method_calls(
        node,
        env_name=env_name,
        fields=method_fields,
        label=f"model '{model_name}' helper '{py_func.__qualname__}'",
    )
    if not changed and not helpers_changed:
        return helper

    _strip_function_annotations(lowered)

    namespace.update(helper_namespace())
    plain = _compile_lowered(
        lowered,
        filename=f"<cimba model '{model_name}' helper "
                f"'{py_func.__qualname__}'>",
        fn_name=py_func.__name__,
        qualname=py_func.__qualname__,
        namespace=namespace,
        like=py_func,
    )
    result = njit(plain)
    cache[key] = result
    return result


def _rewire_entity_method_helpers(
    namespace: dict[str, Any],
    names: Iterable[str],
    *,
    model_name: str,
    method_fields: Mapping[str, str],
    cache: dict[int, Any],
) -> bool:
    """Rewrite, in place, every referenced global name in ``namespace``
    that is a helper dispatcher needing method lowering. Returns
    whether anything changed."""
    changed = False
    for name in names:
        obj = namespace.get(name)
        if obj is None:
            continue
        rewritten = _lower_entity_method_helper(
            obj, model_name=model_name, method_fields=method_fields,
            cache=cache)
        if rewritten is not obj:
            namespace[name] = rewritten
            changed = True
    return changed
