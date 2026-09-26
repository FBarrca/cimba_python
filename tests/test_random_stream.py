"""Ad-hoc draws use the standalone native runtime."""

import cimba as cb


class Routing(cb.Model):
    choice: cb.Output[float]

    @cb.process
    def run(self):
        self.choice = cb.random.uniform() + cb.random.normal()


def test_random_draws_are_seeded_and_worker_invariant():
    model = Routing()
    experiment = cb.Experiment(model, replications=6, seed=718)
    one = experiment.run(workers=1)[model].choice.values
    many = experiment.run(workers=3)[model].choice.values
    assert one.tolist() == many.tolist()
    assert len(set(one[0])) == 6
