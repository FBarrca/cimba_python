"""Cross-instance access uses the declared class view, not a flat namespace."""

import cimba as cb
import pytest
from cimba.diagrams import mermaid


class Worker(cb.Model):
    value: cb.State[float] = 1.0
    total: cb.Output[float]

    @cb.on_end
    def finish(self):
        self.total = self.value


class Team(cb.Model):
    worker: Worker
    selected: cb.Ref[Worker]
    seen: cb.Output[float]

    @cb.on_start
    def configure(self):
        self.worker.value = 3.0
        self.selected.value += 2.0

    @cb.on_end
    def finish(self):
        self.seen = self.selected.value


def test_child_and_reference_views():
    worker = Worker()
    team = Team()
    team.worker = worker
    team.selected = worker
    result = cb.Experiment(team, replications=2).run(workers=1)
    assert result[team].seen.values.tolist() == [[5.0, 5.0]]
    assert result[worker].total.values.tolist() == [[5.0, 5.0]]


class Fleet(cb.Model):
    workers: list[Worker]
    selected: list[cb.Ref[Worker]]
    count: cb.Output[int]

    @cb.on_start
    def configure(self):
        self.count = len(self.workers)
        self.workers[0].value = 4.0
        self.selected[1].value = 6.0


def test_collection_views():
    workers = [Worker(), Worker()]
    fleet = Fleet()
    fleet.workers = workers
    fleet.selected = workers
    result = cb.Experiment(fleet).run(workers=1)
    assert result[fleet].count.values.tolist() == [[2.0]]
    assert result[workers[0]].total.values.tolist() == [[4.0]]
    assert result[workers[1]].total.values.tolist() == [[6.0]]


class BadIndex(cb.Model):
    workers: list[Worker]

    @cb.on_start
    def configure(self):
        self.workers[2].value = 1.0


def test_collection_bounds_abandon_trial():
    model = BadIndex()
    model.workers = [Worker()]
    result = cb.Experiment(model, replications=2).run(workers=1)
    assert result.failed.tolist() == [[True, True]]


class SignedIndex(cb.Model):
    workers: list[Worker]
    chosen: cb.Output[int]
    last: cb.Output[int]
    seen: cb.Output[float]

    @cb.process
    def select(self):
        chosen = -1
        for index in range(len(self.workers)):
            chosen = index
        self.chosen = chosen
        last = len(self.workers) - 1
        self.last = last
        if chosen >= 0:
            self.workers[chosen].value = 7.0
            self.seen = self.workers[last].value


@pytest.mark.parametrize("count", [0, 2])
def test_collection_length_supports_signed_arithmetic_and_sentinel_indices(count):
    model = SignedIndex()
    model.workers = [Worker() for _ in range(count)]
    result = cb.Experiment(model).run(workers=1, on_failure="raise")
    assert result[model].chosen[0, 0] == count - 1
    assert result[model].last[0, 0] == count - 1
    if count:
        assert result[model].seen[0, 0] == 7.0


def test_negative_collection_index_still_abandons_trial():
    class NegativeIndex(cb.Model):
        workers: list[Worker]

        @cb.process
        def select(self):
            self.workers[-1].value = 1.0

    model = NegativeIndex()
    model.workers = [Worker()]
    result = cb.Experiment(model).run(workers=1)
    assert result.failed[0, 0]


class OtherWorker(cb.Model):
    value: cb.State[float] = 2.0
    total: cb.Output[float]

    @cb.on_end
    def finish(self):
        self.total = self.value


class OtherTeam(cb.Model):
    worker: OtherWorker
    selected: cb.Ref[OtherWorker]
    seen: cb.Output[float]

    @cb.on_start
    def configure(self):
        self.worker.value = 8.0

    @cb.on_end
    def finish(self):
        self.seen = self.selected.value


def test_identical_record_shapes_keep_class_specific_views():
    first = Worker()
    team = Team()
    team.worker = first
    team.selected = first
    assert cb.Experiment(team).run(workers=1)[team].seen.values[0, 0] == 5.0
    second = OtherWorker()
    other = OtherTeam()
    other.worker = second
    other.selected = second
    assert cb.Experiment(other).run(workers=1)[other].seen.values[0, 0] == 8.0


class BaseNode(cb.Model):
    base: cb.State[float] = 1.0
    observed: cb.Output[float]

    @cb.on_end
    def finish(self):
        self.observed = self.base


class DerivedNode(BaseNode):
    extra: cb.State[float] = 9.0


class MixedNodes(cb.Model):
    nodes: list[BaseNode]

    @cb.on_start
    def configure(self):
        self.nodes[1].base = 7.0


def test_base_view_over_polymorphic_collection_prefix():
    nodes = [BaseNode(), DerivedNode()]
    model = MixedNodes()
    model.nodes = nodes
    result = cb.Experiment(model).run(workers=1)
    assert result[nodes[1]].observed.values[0, 0] == 7.0


def test_structure_description_and_diagram():
    worker = Worker()
    team = Team()
    team.worker = worker
    team.selected = worker
    description = team.describe()
    assert [item["class"] for item in description] == ["Team", "Worker"]
    diagram = mermaid(team)
    assert "team.worker: Worker" in diagram
    assert "-.->|selected|" in diagram


class OptionalNode(cb.Model):
    upstream: cb.Ref["OptionalNode"] | None
    value: cb.State[int] = 0

    @cb.process
    def act(self):
        if self.upstream is not None:
            self.upstream.value += 1


class OptionalGraph(cb.Model):
    left: OptionalNode
    right: OptionalNode
    total: cb.Output[int]

    @cb.on_end
    def finish(self):
        self.total = self.left.value


def test_optional_reference_is_checked_before_access():
    left = OptionalNode()
    right = OptionalNode()
    right.upstream = left
    graph = OptionalGraph()
    graph.left = left
    graph.right = right
    result = cb.Experiment(graph).run(workers=1)
    assert result[graph].total.values.tolist() == [[1.0]]


def test_incompatible_reference_is_rejected_before_compilation():
    team = Team()
    team.worker = Worker()
    setattr(team, "selected", OtherWorker())
    with pytest.raises(ValueError, match="selected"):
        cb.Experiment(team)
