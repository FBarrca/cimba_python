"""Tutorial 2.1: mice, rats and a cat competing for cheese.

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

"""

import time

import numpy as np

import cimba as cb


NUM_MICE = 5
NUM_RATS = 2
CHEESE_AMOUNT = 20
DURATION = 100_000.0


class Rodent(cb.Model):
    game: cb.Ref["CheeseGame"]
    is_rat: cb.Param[int] = 0
    process_pointer: int = 0

    def __init__(self, game, is_rat):
        self.game = game
        self.is_rat = is_rat

    @cb.process
    def forage(self):
        me = cb.this_process()
        self.process_pointer = me.pointer
        held = 0
        while True:
            if self.is_rat:
                amount = cb.random.dice(3, 10)
                me.priority_set(cb.random.dice(-5, 15))
                signal = self.game.cheese.preempt(amount)
            else:
                amount = cb.random.dice(1, 5)
                me.priority_set(cb.random.dice(-10, 10))
                signal = self.game.cheese.acquire(amount)

            new_held = int(self.game.cheese.held(me))
            if self.is_rat:
                if signal == 0:
                    self.game.rats_grabbed += amount
                elif signal == -1:
                    self.game.rats_preempted += 1
                else:
                    self.game.rats_interrupted += 1
                if held + amount > new_held:
                    self.game.rats_stolen += held + amount - new_held
            else:
                if signal == 0:
                    self.game.mice_grabbed += amount
                elif signal == -1:
                    self.game.mice_preempted += 1
                else:
                    self.game.mice_interrupted += 1
                if held + amount > new_held:
                    self.game.mice_stolen += held + amount - new_held
            held = new_held

            signal = cb.hold(cb.random.exponential(1.0))
            if self.is_rat:
                if signal == -1:
                    self.game.rats_preempted += 1
                elif signal != 0:
                    self.game.rats_interrupted += 1
            else:
                if signal == -1:
                    self.game.mice_preempted += 1
                elif signal != 0:
                    self.game.mice_interrupted += 1
            actual = int(self.game.cheese.held(me))
            if actual < held:
                if self.is_rat:
                    self.game.rats_stolen += held - actual
                else:
                    self.game.mice_stolen += held - actual
            held = actual
            if held > 1:
                drop = cb.random.dice(1, held)
                self.game.cheese.release(drop)
                held -= drop
            if held != self.game.cheese.held(me):
                self.game.accounting_errors += 1
            signal = cb.hold(cb.random.exponential(1.0))
            if signal == -1:
                if self.is_rat:
                    self.game.rats_preempted += 1
                else:
                    self.game.mice_preempted += 1
            actual = int(self.game.cheese.held(me))
            if actual < held:
                if self.is_rat:
                    self.game.rats_stolen += held - actual
                else:
                    self.game.mice_stolen += held - actual
            held = actual


class CheeseGame(cb.Model):
    rodents: list[Rodent]
    cheese: cb.Resource = cb.Resource(capacity=CHEESE_AMOUNT)
    mice_grabbed: cb.Output[int]
    mice_stolen: cb.Output[int]
    mice_preempted: cb.Output[int]
    mice_interrupted: cb.Output[int]
    rats_grabbed: cb.Output[int]
    rats_stolen: cb.Output[int]
    rats_preempted: cb.Output[int]
    rats_interrupted: cb.Output[int]
    cat_chases: cb.Output[int]
    accounting_errors: cb.Output[int]
    cheese_in_use: cb.Output[float]

    def __init__(self):
        self.rodents = ([Rodent(self, 0) for _ in range(NUM_MICE)] +
                        [Rodent(self, 1) for _ in range(NUM_RATS)])

    @cb.process
    def cat(self):
        while True:
            cb.hold(cb.random.exponential(5.0))
            while True:
                cb.hold(cb.random.exponential(1.0))
                index = cb.random.dice(0, NUM_MICE + NUM_RATS - 1)
                pointer = self.rodents[index].process_pointer
                if pointer:
                    signal = -2 if cb.random.bernoulli(0.5) else cb.random.dice(10, 100)
                    cb.Process(pointer).interrupt(signal, 0)
                    self.cat_chases += 1
                if not cb.random.bernoulli(0.5):
                    break

    @cb.on_end
    def game_stats(self):
        self.cheese_in_use = self.cheese.mean_in_use()


def main() -> None:
    game = CheeseGame()
    start = time.perf_counter()
    results = cb.Experiment(
        game, replications=10, window=cb.Window(duration=DURATION),
        seed=20260612,
    ).run()
    elapsed = time.perf_counter() - start
    if results.failed.any():
        raise RuntimeError(f"{results.failed.sum()} cheese trials failed")
    print(f"cimba {cb.engine_version()}; {NUM_MICE} mice, {NUM_RATS} rats, "
          f"{CHEESE_AMOUNT} cheese cubes")
    print(f"{results.failed.size} trials in {elapsed:.2f} s")
    for kind in ("mice", "rats"):
        print(kind, *(f"{getattr(results[game], kind + '_' + name).values.mean():.1f}"
                      for name in ("grabbed", "stolen", "preempted", "interrupted")))
    print(f"cat chases: {results[game].cat_chases.values.mean():.1f}")
    assert np.all(results[game].accounting_errors.values == 0)


if __name__ == "__main__":
    main()
