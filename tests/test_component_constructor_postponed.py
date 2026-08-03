"""Postponed-annotation coverage for Component constructor configuration."""

from __future__ import annotations

import cimba.sim as sim


class PostponedBase(sim.Component):
    baseline: sim.Param = 1.0
    count: sim.Const[int]


class PostponedPolicy(PostponedBase):
    lot_size: sim.Const[float] = 100.0


class PostponedModel(sim.Model):
    policy: PostponedPolicy = PostponedPolicy(count="4", lot_size=250)


def test_component_constructor_resolves_postponed_inherited_annotations():
    policy = PostponedModel.__annotations__["policy"]
    assert policy == "PostponedPolicy"

    model = PostponedModel()
    decl = next(item for item in model._component_decls
                if item.name == "policy")
    assert decl.constants["count"] == (4,)
    assert decl.constants["lot_size"] == (250.0,)
