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


   model = Clinic("Clinic")


   @model.process(copies=3)
   def patient(env: Clinic):
       while True:
           sim.hold(random.exponential(2.0))   # time until this patient returns
           env.doctor.acquire()             # wait for the one free doctor
           sim.hold(random.exponential(1.0))   # consultation
           env.doctor.release()
           env.n_served = env.n_served + 1


   @model.collect
   def collect(env: Clinic):
       env.served = env.n_served


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
