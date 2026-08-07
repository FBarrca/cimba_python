"""Tutorial 4.0: empty harbor simulation template.

This is the Python equivalent of the upstream empty C shell. Fill in a
``sim.Model`` subclass, declare processes with ``@sim.process``, declare
predicates with ``@sim.predicate``, and finish with ``model.experiment(...)``.
"""

import cimba.sim as sim


class HarborTemplate(sim.Model):
    result: sim.Output

    @sim.process
    def placeholder(self):
        self.result = 0.0
        sim.suspend()


model = HarborTemplate()




def main() -> None:
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0, seed=1)
    exp.run()
    print(f"template result: {exp.results.result[0]:.1f}")


if __name__ == "__main__":
    main()
