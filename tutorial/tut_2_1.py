import cimba


def run_preemption_demo() -> list[tuple]:
    log = []

    def mouse(me, resource):
        assert resource.acquire() == cimba.SUCCESS
        log.append(("mouse-acquired", cimba.time(), resource.held_by(me)))
        sig = cimba.hold(10.0)
        log.append(("mouse-hold-returned", cimba.time(), sig, resource.held_by(me)))

    def rat(me, resource):
        cimba.hold(1.0)
        me.priority = 10
        assert resource.preempt() == cimba.SUCCESS
        log.append(("rat-preempted", cimba.time(), resource.held_by(me), resource.available))
        resource.release()

    with cimba.Simulation(seed=21) as sim:
        cheese = cimba.Resource("Cheese")
        cimba.Process("Mouse", mouse, cheese, priority=0).start()
        cimba.Process("Rat", rat, cheese, priority=0).start()
        sim.execute()

    return log


def run_interruption_demo() -> list[tuple]:
    log = []

    def holder(me, cheese):
        assert cheese.acquire(4) == cimba.SUCCESS
        cimba.hold(2.0)
        cheese.release(4)

    def waiting_mouse(me, cheese):
        cimba.hold(0.1)
        sig = cheese.acquire(1)
        log.append(("waiting-mouse", cimba.time(), sig, cheese.held_by(me), cheese.in_use))

    def cat(me, target):
        cimba.hold(0.5)
        target.interrupt(77)

    with cimba.Simulation(seed=22) as sim:
        cheese = cimba.ResourcePool("Cheese", capacity=4)
        cimba.Process("Holder", holder, cheese, priority=0).start()
        target = cimba.Process("WaitingMouse", waiting_mouse, cheese, priority=0).start()
        cimba.Process("Cat", cat, target).start()
        sim.execute()

    return log


def main() -> None:
    print(run_preemption_demo())
    print(run_interruption_demo())


if __name__ == "__main__":
    main()
