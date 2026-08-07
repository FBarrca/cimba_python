Single Server Resource
======================

Use :class:`~cimba.sim.Resource` when a process must acquire exclusive access
to a shared server before it can continue. A process calls ``.acquire()`` to
take the resource — blocking until it is free — and ``.release()`` to hand it
back to the next waiter.

Here three patients repeatedly return to a clinic that has a single doctor.

.. code-block:: python

   import cimba.sim as sim

   import cimba.random as random


   class Clinic(sim.Model):
       served: sim.Output       # patients seen over the run
       n_served: sim.State
       doctor: sim.Resource     # a single shared server

       @sim.process(copies=3)
       def patient(self: "Clinic"):
           while True:
               sim.hold(random.exponential(2.0))
               self.doctor.acquire()
               sim.hold(random.exponential(1.0))
               self.doctor.release()
               self.n_served = self.n_served + 1

       @sim.collect
       def collect(self: "Clinic"):
           self.served = self.n_served


   model = Clinic("Clinic")

   def main() -> None:
       exp = model.experiment(
           replications=1, duration=100.0, warmup=0.0, seed=123
       )
       exp.run()
       print(int(exp["served"][0]))


   if __name__ == "__main__":
       main()

While one patient holds the doctor, the others block in ``.acquire()``;
when it is released the next waiting patient is admitted.
