"""Tests for native Component declaration configuration."""

import pytest

import cimba.sim as sim


def test_component_constructor_configures_inherited_fields_before_flattening():
    class Policy(sim.Component):
        base: sim.Param = 2.0
        count: sim.Const[int]

    class FixedLot(Policy):
        lot_size: sim.Const[float] = 100.0
        observed: sim.Output

        @sim.process
        def report(self, env):
            self.observed = self.lot_size
            sim.suspend()

    configured = FixedLot(base=3, count="4", lot_size=250)

    assert configured.base == 3.0
    assert configured.count == 4
    assert configured.lot_size == 250.0

    class Model(sim.Model):
        policy: FixedLot = configured

    model = Model()
    assert model.param_defaults == {"policy__base": 3.0}
    decl = next(item for item in model._component_decls
                if item.name == "policy")
    assert decl.constants["count"] == (4,)
    assert decl.constants["lot_size"] == (250.0,)
    experiment = model.experiment(replications=1, duration=1.0)
    assert experiment.run() == 0
    assert experiment["policy__observed"][0] == 250.0


def test_component_constructor_coercion_and_param_validation():
    class Values(sim.Component):
        integer: sim.Const[int]
        real: sim.Param

    values = Values(integer="7", real=2)
    assert values.integer == 7
    assert values.real == 2.0

    with pytest.raises(TypeError, match="Param 'real'.*real scalar"):
        Values(real=True)
    with pytest.raises(TypeError, match="Param 'real'.*real scalar"):
        Values(real="2")
    with pytest.raises(TypeError, match="Const 'integer'.*converted to int"):
        Values(integer="not-an-int")


def test_component_constructor_rejects_unknown_and_runtime_fields():
    class Leaf(sim.Component):
        pass

    class RuntimeFields(sim.Component):
        state: sim.State
        output: sim.Output
        queue: sim.Queue
        target: sim.Ref[Leaf]
        child: Leaf

    for name in ("state", "output", "queue", "target", "child"):
        with pytest.raises(TypeError, match=f"field '{name}'.*runtime"):
            RuntimeFields(**{name: object()})

    with pytest.raises(TypeError, match="unexpected keyword argument 'other'"):
        RuntimeFields(other=1)


def test_component_constructor_honors_subclass_annotation_override():
    class Base(sim.Component):
        value: sim.Param

    class Override(Base):
        value: sim.State

    with pytest.raises(TypeError, match="field 'value'.*runtime"):
        Override(value=1)


def test_custom_constructor_forwards_declaration_values():
    class Supplier:
        pass

    source = Supplier()

    class SingleSource(sim.Component):
        rate: sim.Param

        def __init__(self, source: Supplier, **kwargs):
            super().__init__(**kwargs)
            self.source = source

    configured = SingleSource(source, rate=5)
    assert configured.source is source
    assert configured.rate == 5.0


def test_missing_const_still_fails_during_model_construction():
    class Required(sim.Component):
        value: sim.Const[int]

    class Model(sim.Model):
        required: Required = Required()

    with pytest.raises(ValueError, match="constant 'value' must be set"):
        Model()
