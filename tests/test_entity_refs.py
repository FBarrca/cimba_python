"""cb.Ref fields to entities: a reference to an entity a model of the same
tree holds behaves in compiled code exactly as the entity itself."""

import pytest

import cimba as cb
from cimba.schema import ModelDefinitionError


def run(model, **kwargs):
    result = cb.Experiment(model, **kwargs).run(workers=1)
    assert not result.failed.any(), result.failure_reasons
    return result[model]


def value(output):
    return output.values.tolist()[0][0]


class Office(cb.Model):
    clerks: cb.Resource = cb.Resource(capacity=2)


class Desk(cb.Model):
    """Uses its office's clerks through a reference."""
    clerks: cb.Ref[cb.Resource]
    held: cb.Output[int]
    available: cb.Output[int]
    capacity: cb.Output[int]
    mean_in_use: cb.Output[float]
    utilization: cb.Output[float]

    @cb.process
    def work(self):
        self.clerks.acquire()
        cb.hold(5.0)
        self.held = self.clerks.held(cb.this_process())
        self.available = self.clerks.available()
        self.clerks.set_capacity(4)
        self.capacity = self.clerks.capacity()
        cb.hold(5.0)
        self.clerks.release()
        self.mean_in_use = self.clerks.mean_in_use()
        self.utilization = self.clerks.utilization()


class Building(cb.Model):
    office: Office
    desk: Desk
    direct: cb.Output[float]

    @cb.process
    def visitor(self):
        cb.hold(1.0)
        self.office.clerks.acquire()
        self.direct = self.office.clerks.mean_in_use()
        self.office.clerks.release()


def building():
    model = Building()
    model.office = Office()
    model.desk = Desk()
    model.desk.clerks = model.office.clerks
    return model


def test_a_referenced_resource_is_the_resource_itself():
    model = building()
    results = cb.Experiment(model).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    desk = results[model.desk]
    assert value(desk.held) == 1
    assert value(desk.available) == 1
    assert value(desk.capacity) == 4
    # The desk's unit for all of the window; the visitor's for none of it
    assert value(desk.mean_in_use) == pytest.approx(1.0)
    # Capacity 2 for 5, then 4 for 5
    assert value(desk.utilization) == pytest.approx(1.0 / 3.0)
    # The visitor reads the same statistics through the office
    assert value(results[model].direct) == pytest.approx(1.0)


class Optional(cb.Model):
    a: cb.Resource = cb.Resource(capacity=1)
    spare: cb.Ref[cb.Resource] | None
    present: cb.Output[int]
    took: cb.Output[int]

    @cb.process
    def run(self):
        spare = self.spare
        self.present = int(spare is not None)
        if spare is not None:
            self.took = spare.acquire()


@pytest.mark.parametrize("assign", [False, True])
def test_an_optional_entity_reference(assign):
    model = Optional()
    model.spare = model.a if assign else None
    out = run(model)
    assert value(out.present) == int(assign)
    if assign:
        assert value(out.took) == 0


class Switch(cb.Model):
    """Model code points the reference at another resource at run time."""
    a: cb.Resource = cb.Resource(capacity=1)
    b: cb.Resource = cb.Resource(capacity=3)
    current: cb.Ref[cb.Resource]
    before: cb.Output[int]
    after: cb.Output[int]

    @cb.process
    def run(self):
        self.before = self.current.capacity()
        self.current = self.b
        self.after = self.current.capacity()


def test_model_code_assigns_an_entity_reference():
    model = Switch()
    model.current = model.a
    out = run(model)
    assert value(out.before) == 1
    assert value(out.after) == 3


class Kinds(cb.Model):
    """Other entity kinds fall out of the same mechanism."""
    tank: cb.Container = cb.Container(initial=5)
    queue: cb.Store[int] = cb.Store()
    stats: cb.Dataset = cb.Dataset()
    gate: cb.Condition = cb.Condition()
    tanks: list[cb.Ref[cb.Container]]
    q: cb.Ref[cb.Store[int]]
    data: cb.Ref[cb.Dataset]
    door: cb.Ref[cb.Condition]
    open: int = 0
    level: cb.Output[int]
    item: cb.Output[int]
    mean: cb.Output[float]
    opened_at: cb.Output[float]

    @cb.predicate
    def is_open(self) -> bool:
        return self.open == 1

    @cb.process
    def use(self):
        self.tanks[0].get(2)
        self.level = self.tank.level()
        self.q.put(7)
        self.item = self.queue.get()
        self.data.record(3.0)
        self.stats.record(5.0)
        self.mean = self.data.sample_mean()
        self.door.wait_until(self.is_open)
        self.opened_at = cb.now()

    @cb.process
    def opener(self):
        cb.hold(2.0)
        self.open = 1
        self.gate.signal()


def test_other_entity_kinds_can_be_referenced():
    model = Kinds()
    model.tanks = [model.tank]
    model.q, model.data, model.door = model.queue, model.stats, model.gate
    out = run(model)
    assert value(out.level) == 3
    assert value(out.item) == 7
    assert value(out.mean) == 4.0
    assert value(out.opened_at) == 2.0


class Visit(cb.Model):
    clerks: cb.Ref[cb.Resource]
    at: cb.Output[float]

    @cb.process
    def visit(self):
        self.clerks.acquire()
        cb.hold(2.0)
        self.clerks.release()
        cb.release(self)


class Spawner(cb.Model):
    clerks: cb.Resource = cb.Resource(capacity=1)
    last: cb.Output[float]

    @cb.process
    def arrivals(self):
        for _ in range(3):
            cb.spawn(Visit, clerks=self.clerks)
        cb.hold(1.0)
        self.clerks.acquire()
        self.last = cb.now()


def test_a_spawned_model_takes_an_entity_reference():
    out = run(Spawner())
    # Three visits of 2 each, queued ahead of the spawner (t=1)
    assert value(out.last) == 6.0


class WrongSpawn(cb.Model):
    tank: cb.Container = cb.Container()

    @cb.process
    def arrivals(self):
        cb.spawn(Visit, clerks=self.tank)


def test_a_spawn_reference_of_the_wrong_kind_is_a_compile_error():
    with pytest.raises(Exception, match="expected a Resource reference"):
        cb.Experiment(WrongSpawn()).run(workers=1)


# ------------------------------------------------------------- host validation

class Holder(cb.Model):
    res: cb.Ref[cb.Resource]
    pool: list[cb.Ref[cb.Resource]]


class Tree(cb.Model):
    a: cb.Resource = cb.Resource()
    tank: cb.Container = cb.Container()
    holder: Holder


def tree(res=None, pool=None):
    model = Tree()
    model.holder = Holder()
    model.holder.res = model.a if res is None else res
    model.holder.pool = [model.a] if pool is None else pool
    return model


def test_a_resource_outside_the_tree_is_rejected():
    stranger = Tree()
    with pytest.raises(ModelDefinitionError, match=r"tree\.holder\.res: references a Resource "
                                                   "that is not an entity of this model tree"):
        cb.Experiment(tree(res=stranger.a))
    with pytest.raises(ModelDefinitionError, match=r"tree\.holder\.pool\[0\]: references"):
        cb.Experiment(tree(pool=[stranger.a]))


def test_an_entity_of_another_kind_is_rejected():
    model = Tree()
    with pytest.raises(ModelDefinitionError, match=r"holder\.res: expected a Resource of this "
                                                   "model tree, got <cimba.*Container"):
        cb.Experiment(tree(res=model.tank))
    model = tree()
    model.holder.pool = [model.a, model.tank]
    with pytest.raises(ModelDefinitionError, match=r"holder\.pool\[1\]: expected a Resource"):
        cb.Experiment(model)


@pytest.mark.parametrize("bad", [3, None, Holder()])
def test_a_value_that_is_no_entity_is_rejected(bad):
    with pytest.raises(ModelDefinitionError, match=r"holder\.pool\[1\]: expected a Resource"):
        model = tree()
        model.holder.pool = [model.a, bad]
        cb.Experiment(model)
    if bad is not None:
        model = tree()
        model.holder.res = bad
        with pytest.raises(ModelDefinitionError, match=r"holder\.res: expected a Resource"):
            cb.Experiment(model)


def test_a_missing_entity_reference_is_rejected():
    model = tree()
    model.holder.res = None
    with pytest.raises(ModelDefinitionError, match=r"holder\.res: reference missing"):
        cb.Experiment(model)


class Typed(cb.Model):
    floats: cb.Store[float] = cb.Store()
    ints: cb.Ref[cb.Store[int]]


def test_a_store_of_another_value_type_is_rejected():
    model = Typed()
    model.ints = model.floats
    with pytest.raises(ModelDefinitionError, match=r"typed\.floats, a Store\[float\], "
                                                   r"not a Store\[int\]"):
        cb.Experiment(model)


# ------------------------------------------------------------------ sweeps

class Line(cb.Model):
    """One swept option: its own pool over the root's roles."""
    pool: list[cb.Ref[cb.Resource]]
    attached_to: cb.Ref["Plant"]

    def attached(self, owner, field):
        self.attached_to = owner


class Plant(cb.Model):
    a: cb.Resource = cb.Resource(capacity=0)
    b: cb.Resource = cb.Resource(capacity=0)
    c: cb.Resource = cb.Resource(capacity=1)
    line: Line
    size: cb.Output[int]
    got: cb.Output[int]

    @cb.process
    def run(self):
        self.size = len(self.line.pool)
        self.got = cb.acquire_any(self.line.pool)


def test_swept_options_hold_pools_of_different_lengths():
    model = Plant()
    short, long = Line(), Line()
    model.line = cb.sweep([short, long])
    short.pool = [model.c]
    long.pool = [model.a, model.b, model.c]
    results = cb.Experiment(model).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[model].size.values.ravel().tolist() == [1, 3]
    assert results[model].got.values.ravel().tolist() == [0, 2]


class Site(cb.Model):
    crew: cb.Resource = cb.Resource(capacity=1)


class Sites(cb.Model):
    site: Site
    crew: cb.Ref[cb.Resource]
    got: cb.Output[int]

    @cb.process
    def run(self):
        self.got = self.crew.acquire()


def test_a_reference_into_an_unselected_option_is_rejected():
    model = Sites()
    first, second = Site(), Site()
    model.site = cb.sweep([first, second])
    model.crew = first.crew
    # The option holding the referenced entity is absent from the second
    # option's tree, as for a model reference
    with pytest.raises(ModelDefinitionError, match="not an entity of this model tree"):
        cb.Experiment(model).run(workers=1)


# ---------------------------------------------------------------- diagrams

def test_structure_draws_entity_references_to_their_holders():
    from cimba.diagrams import structure

    edges = {(e.source, e.target, e.label, e.style) for e in structure(building()).edges}
    assert ("building.desk", "building.office", "clerks → clerks", "dotted") in edges


def test_process_graph_follows_entity_references():
    from cimba.diagrams import process_graph

    edges = {(e.source, e.label, e.target) for e in process_graph(building()).edges}
    assert ("building.office.clerks", "acquire", "building.desk.work") in edges
    assert ("building.desk.work", "release", "building.office.clerks") in edges
    spawned = {(e.source, e.label, e.target) for e in process_graph(Spawner()).edges}
    assert ("spawner.clerks", "acquire", "Visit.visit") in spawned


class WrongAssignment(cb.Model):
    a: cb.Resource = cb.Resource()
    tank: cb.Container = cb.Container()
    current: cb.Ref[cb.Resource]

    @cb.process
    def run(self):
        self.current = self.tank


def test_assigning_an_entity_of_another_kind_is_a_compile_error():
    model = WrongAssignment()
    model.current = model.a
    with pytest.raises(Exception, match="cannot use a Container where a Resource is expected"):
        cb.Experiment(model).run(workers=1)
