"""cb.acquire_any over a list[cb.Ref[cb.Resource]]: the members are
configuration, sized at run time, with the semantics of the tuple form (see
test_acquire_any.py). Exact timestamps throughout."""

import pytest

import cimba as cb

PREEMPTED = -1
INTERRUPTED = -2
TIMEOUT = -5


def run(model, **kwargs):
    result = cb.Experiment(model, **kwargs).run(workers=1)
    assert not result.failed.any(), result.failure_reasons
    return result[model]


def value(output):
    return output.values.tolist()[0][0]


# ----------------------------------------------------------- the issue's case

class Role(cb.Model):
    res: cb.Resource = cb.Resource(capacity=1)


class Pool(cb.Model):
    resources: list[cb.Ref[cb.Resource]]


class Root(cb.Model):
    """Five tasks, each staffed by any one role of the pool; a role works a
    task for 4. Roles 0..2 are in the pool, role 3 is not."""
    roles: list[Role]
    pool: Pool
    finished: cb.Output[float]

    @cb.process(copies=5)
    def task(self) -> None:
        granted = cb.acquire_any(self.pool.resources, priority=0)
        cb.hold(4.0)
        self.pool.resources[granted].release()
        self.finished = cb.now()


def staffed(roles=4, members=(0, 1, 2)):
    root = Root()
    root.roles = [Role() for _ in range(roles)]
    root.pool = Pool()
    root.pool.resources = [root.roles[i].res for i in members]
    return root


def test_the_issue_repro_runs_with_a_configured_pool():
    root = staffed()
    out = run(root)
    # Three roles serve five tasks: two waves of 4
    assert value(out.finished) == 8.0


# ------------------------------------------------------------ semantics, ported

class Desk(cb.Model):
    """As test_acquire_any.Desk, with the pair configured as a list."""
    clerks: cb.Resource = cb.Resource(capacity=1)
    seniors: cb.Resource = cb.Resource(capacity=1)
    pair: list[cb.Ref[cb.Resource]]
    any_arrives: cb.Param[float] = 1.0
    direct_arrives: cb.Param[float] = 2.0
    any_priority: cb.Param[int] = -1
    any_got: cb.Output[int]
    any_at: cb.Output[float]
    direct_at: cb.Output[float]
    later_at: cb.Output[float]

    @cb.process(priority=9)
    def holders(self):
        self.clerks.acquire()
        self.seniors.acquire()
        cb.hold(10.0)
        self.clerks.release()
        cb.hold(2.0)
        self.seniors.release()

    @cb.process
    def any(self):
        cb.hold(self.any_arrives)
        if self.any_priority < 0:
            self.any_got = cb.acquire_any(self.pair)
        else:
            self.any_got = cb.acquire_any(self.pair, priority=self.any_priority)
        self.any_at = cb.now()
        cb.hold(5.0)
        self.pair[self.any_got].release()

    @cb.process
    def direct(self):
        cb.hold(self.direct_arrives)
        self.clerks.acquire()
        self.direct_at = cb.now()
        cb.hold(5.0)
        self.clerks.release()

    @cb.process
    def later(self):
        cb.hold(3.0)
        self.seniors.acquire()
        self.later_at = cb.now()
        self.seniors.release()


def desk():
    model = Desk()
    model.pair = [model.clerks, model.seniors]
    return model


def test_earlier_acquire_any_is_served_first_and_withdrawn_elsewhere():
    out = run(desk())
    assert value(out.any_got) == 0
    assert value(out.any_at) == 10.0
    assert value(out.direct_at) == 15.0
    assert value(out.later_at) == 12.0


def test_earlier_direct_acquirer_is_served_first():
    model = desk()
    model.any_arrives, model.direct_arrives = 2.0, 1.0
    out = run(model)
    assert value(out.direct_at) == 10.0
    assert value(out.any_got) == 1
    assert value(out.any_at) == 12.0
    assert value(out.later_at) == 17.0


def test_explicit_priority_goes_ahead_of_earlier_arrivals():
    model = desk()
    model.any_arrives, model.direct_arrives = 2.0, 1.0
    model.any_priority = 3
    out = run(model)
    assert value(out.any_got) == 0
    assert value(out.any_at) == 10.0
    assert value(out.direct_at) == 15.0
    assert value(out.later_at) == 12.0


class Free(cb.Model):
    a: cb.Resource = cb.Resource(capacity=1)
    b: cb.Resource = cb.Resource(capacity=1)
    ba: list[cb.Ref[cb.Resource]]
    aba: list[cb.Ref[cb.Resource]]
    first: cb.Output[int]
    second: cb.Output[int]
    third: cb.Output[int]

    @cb.process
    def take(self):
        self.first = cb.acquire_any(self.ba)    # both free: b, listed first
        self.second = cb.acquire_any(self.ba)   # only a is free
        self.b.release()
        self.third = cb.acquire_any(self.aba)   # duplicates: b is index 1


def test_first_listed_free_resource_wins_and_duplicates_collapse():
    model = Free()
    model.ba = [model.b, model.a]
    model.aba = [model.a, model.b, model.a]
    out = run(model)
    assert value(out.first) == 0
    assert value(out.second) == 1
    assert value(out.third) == 1


class Twice(cb.Model):
    """A list naming one resource twice waits once in its queue: the grant
    at t=5 returns the first index, and Direct (t=2) is served after it."""
    a: cb.Resource = cb.Resource(capacity=1)
    twice: list[cb.Ref[cb.Resource]]
    got: cb.Output[int]
    at: cb.Output[float]
    direct_at: cb.Output[float]

    @cb.process(priority=9)
    def holder(self):
        self.a.acquire()
        cb.hold(5.0)
        self.a.release()

    @cb.process
    def any(self):
        cb.hold(1.0)
        self.got = cb.acquire_any(self.twice)
        self.at = cb.now()
        cb.hold(1.0)
        self.a.release()

    @cb.process
    def direct(self):
        cb.hold(2.0)
        self.a.acquire()
        self.direct_at = cb.now()
        self.a.release()


def test_a_duplicated_member_waits_once_and_returns_the_first_index():
    model = Twice()
    model.twice = [model.a, model.a]
    out = run(model)
    assert value(out.got) == 0
    assert value(out.at) == 5.0
    assert value(out.direct_at) == 6.0


class StaleAny(cb.Model):
    clerks: cb.Resource = cb.Resource(capacity=1)
    seniors: cb.Resource = cb.Resource(capacity=1)
    pair: list[cb.Ref[cb.Resource]]
    any_pointer: int = 0
    when: cb.Param[float] = 10.0
    any_signal: cb.Output[int]
    any_held: cb.Output[int]
    any_back: cb.Output[float]
    direct_at: cb.Output[float]
    later_at: cb.Output[float]

    @cb.process(priority=5)
    def holder(self):
        self.clerks.acquire()
        self.seniors.acquire()
        cb.hold(10.0)
        self.clerks.release()
        cb.hold(2.0)
        self.seniors.release()

    @cb.process
    def any(self):
        self.any_pointer = cb.this_process().pointer
        cb.hold(1.0)
        self.any_signal = cb.acquire_any(self.pair)
        me = cb.this_process()
        self.any_held = self.clerks.held(me) + self.seniors.held(me)
        self.any_back = cb.now()
        cb.hold(100.0)

    @cb.process
    def direct(self):
        cb.hold(2.0)
        self.clerks.acquire()
        self.direct_at = cb.now()
        cb.hold(1.0)
        self.clerks.release()

    @cb.process
    def later(self):
        cb.hold(3.0)
        self.seniors.acquire()
        self.later_at = cb.now()
        self.seniors.release()

    @cb.process(priority=3)
    def cat(self):
        cb.hold(self.when)
        cb.Process(self.any_pointer).interrupt(INTERRUPTED, 4)


def stale(when):
    model = StaleAny()
    model.pair = [model.clerks, model.seniors]
    model.when = when
    return model


@pytest.mark.parametrize("when, back", [(10.0, 10.0), (4.0, 4.0)])
def test_interrupt_leaves_no_stale_entries(when, back):
    out = run(stale(when))
    assert value(out.any_signal) == INTERRUPTED
    assert value(out.any_held) == 0
    assert value(out.any_back) == back
    assert value(out.direct_at) == 10.0
    assert value(out.later_at) == 12.0


class Bumped(cb.Model):
    clerks: cb.Resource = cb.Resource(capacity=1)
    seniors: cb.Resource = cb.Resource(capacity=0)
    pair: list[cb.Ref[cb.Resource]]
    got: cb.Output[int]
    hold_signal: cb.Output[int]
    held_after: cb.Output[int]
    boss_at: cb.Output[float]

    @cb.process
    def any(self):
        self.got = cb.acquire_any(self.pair)
        self.hold_signal = cb.hold(10.0)
        self.held_after = self.clerks.held(cb.this_process())

    @cb.process(priority=5)
    def boss(self):
        cb.hold(2.0)
        self.clerks.preempt()
        self.boss_at = cb.now()
        self.clerks.release()


def test_an_acquire_any_holding_can_be_preempted():
    model = Bumped()
    model.pair = [model.clerks, model.seniors]
    out = run(model)
    assert value(out.got) == 0
    assert value(out.hold_signal) == PREEMPTED
    assert value(out.held_after) == 0
    assert value(out.boss_at) == 2.0


class Staffed(cb.Model):
    clerks: cb.Resource = cb.Resource(capacity=0)
    seniors: cb.Resource = cb.Resource(capacity=0)
    pair: list[cb.Ref[cb.Resource]]
    got: cb.Output[int]
    at: cb.Output[float]

    @cb.process
    def any(self):
        cb.hold(1.0)
        self.got = cb.acquire_any(self.pair)
        self.at = cb.now()
        self.seniors.release()

    @cb.process
    def staffing(self):
        cb.hold(4.0)
        self.pair[1].set_capacity(1)


def test_raising_capacity_grants_a_waiting_acquire_any():
    model = Staffed()
    model.pair = [model.clerks, model.seniors]
    out = run(model)
    assert value(out.got) == 1
    assert value(out.at) == 4.0


class TimedOut(cb.Model):
    clerks: cb.Resource = cb.Resource(capacity=1)
    seniors: cb.Resource = cb.Resource(capacity=0)
    pair: list[cb.Ref[cb.Resource]]
    signal: cb.Output[int]
    at: cb.Output[float]
    direct_at: cb.Output[float]

    @cb.process(priority=9)
    def holder(self):
        self.clerks.acquire()
        cb.hold(10.0)
        self.clerks.release()

    @cb.process
    def any(self):
        cb.hold(1.0)
        cb.this_process().timer_set(3.0, TIMEOUT)
        self.signal = cb.acquire_any(self.pair)
        self.at = cb.now()

    @cb.process
    def direct(self):
        cb.hold(2.0)
        self.clerks.acquire()
        self.direct_at = cb.now()


def test_a_process_timer_bounds_the_wait():
    model = TimedOut()
    model.pair = [model.clerks, model.seniors]
    out = run(model)
    assert value(out.signal) == TIMEOUT
    assert value(out.at) == 4.0
    assert value(out.direct_at) == 10.0


class Crowd(cb.Model):
    a: cb.Resource = cb.Resource(capacity=1)
    b: cb.Resource = cb.Resource(capacity=1)
    ab: list[cb.Ref[cb.Resource]]
    ba: list[cb.Ref[cb.Resource]]
    bad: cb.Param[int] = 1

    @cb.process(copies=3)
    def holder(self):
        cb.acquire_any(self.ab)
        cb.hold(100.0)

    @cb.process
    def ranked(self):
        cb.hold(1.0)
        cb.acquire_any(self.ba, priority=4)

    @cb.process
    def culprit(self):
        cb.hold(3.0)
        if self.bad:
            self.a.release(1)


def test_abandoned_trials_with_acquire_any_waiters_leak_nothing():
    model = Crowd()
    model.ab, model.ba = [model.a, model.b], [model.b, model.a]
    results = cb.Experiment(model, replications=200).run(workers=1)
    assert results.failed.all()
    assert "release(1) exceeds the units held" in results.failure_reasons[0][0]
    model.bad = 0
    assert not cb.Experiment(model, replications=50,
                             window=cb.Window(duration=10)).run(workers=1).failed.any()


# ----------------------------------------------------------- run-time sizes

class Big(cb.Model):
    """Pools of different sizes, all served by one compiled method. Every
    member is busy until its release at t = 1 + index; the pool's waiter
    gets the first member to free up, the last listed."""
    roles: list[Role]
    pools: list[Pool]
    got: cb.Output[int] = 0
    at: cb.Output[float] = 0.0
    sizes: cb.Output[int] = 0

    @cb.process(priority=9)
    def holders(self):
        for i in range(len(self.roles)):
            self.roles[i].res.acquire()
        for i in range(len(self.roles)):
            cb.hold(1.0)
            self.roles[len(self.roles) - 1 - i].res.release()

    @cb.process
    def waiters(self):
        cb.hold(0.5)
        for p in range(len(self.pools)):
            members = self.pools[p].resources
            self.sizes = self.sizes * 100 + len(members)
            granted = cb.acquire_any(members)
            self.got = self.got * 100 + granted
            self.at = cb.now()


def test_large_pools_and_sizes_chosen_at_run_time():
    model = Big()
    model.roles = [Role() for _ in range(20)]
    big, small = Pool(), Pool()
    big.resources = [role.res for role in model.roles]            # 20 members
    small.resources = [model.roles[i].res for i in (17, 18)]       # 2 members
    model.pools = [big, small]
    out = run(model)
    assert value(out.sizes) == 2002
    # The holder frees role 19 at t=1 (big's index 19), then 18 at t=2
    # (small's index 1)
    assert value(out.got) == 1901
    assert value(out.at) == 2.0


class Empty(cb.Model):
    pool: list[cb.Ref[cb.Resource]]

    @cb.process
    def take(self):
        cb.acquire_any(self.pool)


def test_an_empty_list_fails_the_trial():
    model = Empty()
    model.pool = []
    results = cb.Experiment(model).run(workers=1)
    assert results.failed.all()
    assert "acquire_any() needs at least one resource" in results.failure_reasons[0][0]


class OutOfRange(cb.Model):
    a: cb.Resource = cb.Resource(capacity=1)
    pool: list[cb.Ref[cb.Resource]]

    @cb.process
    def take(self):
        self.pool[1].acquire()


def test_an_index_out_of_range_fails_the_trial():
    model = OutOfRange()
    model.pool = [model.a]
    assert cb.Experiment(model).run(workers=1).failed.all()


class Members(cb.Model):
    roles: list[Role]
    members: list[cb.Ref[Role]]

    @cb.process
    def take(self):
        cb.acquire_any(self.members)


def test_a_list_of_models_is_not_a_list_of_resources():
    model = Members()
    model.roles = [Role()]
    model.members = list(model.roles)
    with pytest.raises(Exception, match=r"list\[cb.Ref\[cb.Resource\]\]"):
        cb.Experiment(model).run(workers=1)


# ------------------------------------------------------------------- diagrams

def test_process_graph_draws_acquire_any_from_every_referenced_member():
    from cimba.diagrams import process_graph

    root = staffed()
    edges = {(e.source, e.label, e.target) for e in process_graph(root).edges}
    for i in range(3):
        assert (f"root.roles[{i}].res", "acquire any", "root.task") in edges
        assert (f"root.roles[{i}].res", "release", "root.task") not in edges
    assert ("root.task", "release", f"root.roles[0].res") in edges
    assert not any(source == "root.roles[3].res" for source, _, _ in edges)
