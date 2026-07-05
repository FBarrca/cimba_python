"""
Mice, rats, and a cat - tutorial 2.1 (subprojects/cimba/tutorial/tut_2_1.c)
through the Python bindings: interrupt and preempt process interactions.

Five mice and two rats compete for a pool of 20 cheese cubes. Each animal
repeatedly picks a random amount and a random priority, then tries to take
the cheese: mice acquire politely (blocking until available), rats preempt
(snatching from lower-priority holders). A preempted animal loses ALL its
holdings, and learns this from the PREEMPTED signal returned by whichever
blocking call it was sitting in. Meanwhile a cat naps and wakes at random
to chase rodents, interrupting them with either the generic INTERRUPTED
signal or a random user-defined signal in [10, 100]; an interrupted call
returns early but holdings are unchanged.

Translation notes (C -> cimba.sim):

* The cat picks its victim through the declared sim.Processes fields
  (env.mouse[i] / env.rat[i]), which publish the handles of each
  @model.process copy - the C version reads the same handles out of its
  simulation struct.
* The C version verifies cheese accounting with debug asserts against
  cmb_resourcepool_held_by_process() after every step; here the checks
  count mismatches into the accounting_errors output (asserted to be 0),
  via ``env.cheese.held()``.
* The C tutorial infers its holdings from the returned signals: PREEMPTED
  means "all my cheese is gone", anything else "unchanged". Both readings
  are wrong: a preemptor only takes what it needs starting from the
  lowest-priority holder (a victim may keep a remainder), a process
  waiting in acquire can have its prior holdings raided yet still return
  SUCCESS once its request is eventually granted, and a PREEMPTED return
  can even leave the victim holding MORE than before (part of the
  in-flight request granted and kept before the raid). The header warns
  "do not assume" -- signals say why you woke, not what you hold -- and
  the C tutorial's own debug assert fails within ~1000 time units. This
  port treats the pool's books (``env.cheese.held()``) as authoritative after
  every blocking call and classifies the net change as grabbed/stolen.
* The C version logs every move; here the events are tallied into
  per-species counters instead.
* C gives each animal a random initial priority at creation; here all
  start at 0 and draw a random priority at the top of the first loop,
  which is equivalent from the second event on.

Usage: uv run python examples/demo_cheese.py
"""

import time

import numpy as np
from numba import njit

import cimba as cp
import cimba.random as random
import cimba.sim as sim

NUM_MICE = 5
NUM_RATS = 2
CHEESE_AMOUNT = 20

DURATION = 100000.0


class CheeseGame(sim.Model):
    # Results
    mice_grabbed: sim.Output        # cheese units successfully acquired
    mice_stolen: sim.Output         # units taken from mice by preemptors
    mice_preempted: sim.Output      # preemption events suffered
    mice_interrupted: sim.Output    # times a mouse was interrupted
    rats_grabbed: sim.Output
    rats_stolen: sim.Output
    rats_preempted: sim.Output
    rats_interrupted: sim.Output
    cat_chases: sim.Output
    accounting_errors: sim.Output   # held-amount mismatches (must be 0)
    cheese_in_use: sim.Output       # time-weighted mean units held

    # Counters (auto-zeroed per trial)
    m_grab: sim.State
    m_stol: sim.State
    m_pre: sim.State
    m_int: sim.State
    r_grab: sim.State
    r_stol: sim.State
    r_pre: sim.State
    r_int: sim.State
    chases: sim.State
    acct_errors: sim.State

    # The pile of cheese cubes
    cheese: sim.Pool = CHEESE_AMOUNT

    # Process handles, used by the cat to pick a victim
    mouse: sim.Processes
    rat: sim.Processes


game = CheeseGame()


@njit
def _take_stock(env, me, expected):
    """Resync our belief with the pool's books after a blocking call.
    Holdings can only shrink while we are blocked (preemptors raiding
    them); anything else is an accounting error. Returns (held, stolen)."""
    held = env.cheese.held(me)
    stolen = expected - held
    if stolen < 0:
        env.acct_errors = env.acct_errors + 1
        stolen = 0
    return held, stolen


@njit
def _forage_once(env, me, preempting, amt_lo, amt_hi, pri_lo, pri_hi, held):
    """One forage cycle of a rodent: take, hold, drop some, rest.
    Returns (held, grabbed, stolen, preempted, interrupted) for this
    cycle; accounting mismatches are counted into env.acct_errors."""
    preempted = 0
    interrupted = 0

    # Decide on a random amount and a random priority for this round
    amount = random.dice(amt_lo, amt_hi)
    sim.set_priority(me, random.dice(pri_lo, pri_hi))
    if preempting == 1:
        sig = env.cheese.preempt(amount)
    else:
        sig = env.cheese.acquire(amount)
    held_now = env.cheese.held(me)
    grabbed = 0
    stolen = 0
    if sig == sim.SUCCESS:
        # The full request was granted, though prior holdings may have
        # been raided while we waited; more than held + amount is a bug
        grabbed = amount
        stolen = held + amount - held_now
        if stolen < 0:
            env.acct_errors = env.acct_errors + 1
            stolen = 0
    else:
        # Preempted or interrupted mid-acquire: part of the request may
        # have been granted and kept, prior holdings may have been
        # raided -- only the net change is knowable
        if sig == sim.PREEMPTED:
            preempted = preempted + 1
        else:
            interrupted = interrupted + 1
        if held_now > held:
            grabbed = held_now - held
        else:
            stolen = held - held_now
    held = held_now

    # Hold on to it for a while
    sig = sim.hold(random.exponential(1.0))
    if sig == sim.PREEMPTED:
        preempted = preempted + 1
    elif sig != sim.SUCCESS:
        interrupted = interrupted + 1
    held, lost = _take_stock(env, me, held)
    stolen = stolen + lost

    # Drop some amount. Release is immediate and exact, so here the
    # books must match our belief to the unit.
    if held > 1:
        release = random.dice(1, held)
        env.cheese.release(release)
        held = held - release
    if held != env.cheese.held(me):
        env.acct_errors = env.acct_errors + 1

    # Hang on a moment before trying again
    sig = sim.hold(random.exponential(1.0))
    if sig == sim.PREEMPTED:
        preempted = preempted + 1
    held, lost = _take_stock(env, me, held)
    stolen = stolen + lost
    return held, grabbed, stolen, preempted, interrupted


@game.process(copies=NUM_MICE)
def mouse(env: CheeseGame):
    me = sim.current()
    held = 0
    while True:
        held, grabbed, stolen, preempted, interrupted = _forage_once(
            env, me, 0, 1, 5, -10, 10, held)
        env.m_grab = env.m_grab + grabbed
        env.m_stol = env.m_stol + stolen
        env.m_pre = env.m_pre + preempted
        env.m_int = env.m_int + interrupted


@game.process(copies=NUM_RATS)
def rat(env: CheeseGame):
    me = sim.current()
    held = 0
    while True:
        held, grabbed, stolen, preempted, interrupted = _forage_once(
            env, me, 1, 3, 10, -5, 15, held)
        env.r_grab = env.r_grab + grabbed
        env.r_stol = env.r_stol + stolen
        env.r_pre = env.r_pre + preempted
        env.r_int = env.r_int + interrupted


@game.process
def cat(env: CheeseGame):
    while True:
        # Nobody interrupts a sleeping cat, disregard the signal
        sim.hold(random.exponential(5.0))
        while True:
            # Awake, looking for rodents
            sim.hold(random.exponential(1.0))
            i = random.dice(0, NUM_MICE + NUM_RATS - 1)
            if i < NUM_MICE:
                target = env.mouse[i]
            else:
                target = env.rat[i - NUM_MICE]
            # Send it the generic signal or a random user-defined one
            if random.bernoulli(0.5) == 1:
                sim.interrupt(target, sim.INTERRUPTED, 0)
            else:
                sim.interrupt(target, random.dice(10, 100), 0)
            env.chases = env.chases + 1
            # Flip a coin to decide whether to go back to sleep
            if random.bernoulli(0.5) == 0:
                break


@game.collect
def game_stats(env: CheeseGame):
    env.mice_grabbed = env.m_grab
    env.mice_stolen = env.m_stol
    env.mice_preempted = env.m_pre
    env.mice_interrupted = env.m_int
    env.rats_grabbed = env.r_grab
    env.rats_stolen = env.r_stol
    env.rats_preempted = env.r_pre
    env.rats_interrupted = env.r_int
    env.cat_chases = env.chases
    env.accounting_errors = env.acct_errors
    env.cheese_in_use = env.cheese.mean_in_use()


def main() -> None:
    print(f"cimba {cp.version()}, using {cp.use_threads(0)} worker threads")
    print(f"{NUM_MICE} mice and {NUM_RATS} rats compete for "
          f"{CHEESE_AMOUNT} cheese cubes, 1 cat chases the rodents")

    exp = game.experiment(replications=10, duration=DURATION, warmup=0.0,
                          seed=20260612)
    t0 = time.perf_counter()
    fails = exp.run()
    wall = time.perf_counter() - t0
    print(f"{len(exp)} trials of {DURATION:.0f} time units in "
          f"{wall:.2f} s, {fails} failed\n")

    def avg(field: str) -> float:
        return float(np.mean(exp[field]))

    print(f"{'':>10} {'grabbed':>10} {'stolen':>10} {'preempted':>10} "
          f"{'interrupted':>11}")
    print(f"{'mice':>10} {avg('mice_grabbed'):10.0f} "
          f"{avg('mice_stolen'):10.0f} "
          f"{avg('mice_preempted'):10.0f} {avg('mice_interrupted'):11.0f}")
    print(f"{'rats':>10} {avg('rats_grabbed'):10.0f} "
          f"{avg('rats_stolen'):10.0f} "
          f"{avg('rats_preempted'):10.0f} {avg('rats_interrupted'):11.0f}")

    print(f"\ncat chases: {avg('cat_chases'):.0f} per trial")
    print(f"cheese in use: {avg('cheese_in_use'):.1f} of "
          f"{CHEESE_AMOUNT} cubes on average")

    errors = int(exp["accounting_errors"].sum())
    print(f"accounting errors (held vs pool_held): {errors}")
    assert errors == 0, "cheese accounting mismatch!"


if __name__ == "__main__":
    main()
