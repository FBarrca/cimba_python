import cimba as cb
from tutorial.tut_5_1 import AssemblyLine, RANDOM_SEED


def test_assembly_line_handoffs_and_bottleneck():
    line = AssemblyLine(duration=500)
    results = cb.Experiment(
        line, window=cb.Window(duration=500), seed=RANDOM_SEED,
    ).run()
    assert not results.failed.any()
    assert results[line].total_parts_produced[0, 0] > 0
    assert results[line].avg_cycle_time[0, 0] > 0
    assert (results[line.station_2].utilization[0, 0] >
            results[line.station_3].utilization[0, 0])
