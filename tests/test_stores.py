"""Scalar queue handles use the engine's FIFO and priority queues."""

import cimba as cb


class Queues(cb.Model):
    fifo: cb.Store[float]
    ranked: cb.PriorityStore[int]
    first_fifo: cb.Output[float]
    first_ranked: cb.Output[int]

    @cb.process(priority=10)
    def producer(self):
        self.fifo.put(1.25)
        self.fifo.put(2.5)
        self.ranked.put(3, priority=1)
        self.ranked.put(7, priority=5)

    @cb.process
    def consumer(self):
        self.first_fifo = self.fifo.get()
        self.first_ranked = self.ranked.get()


def test_scalar_stores():
    queues = Queues()
    result = cb.Experiment(queues, replications=2).run(workers=1)
    assert result[queues].first_fifo.values.tolist() == [[1.25, 1.25]]
    assert result[queues].first_ranked.values.tolist() == [[7.0, 7.0]]
