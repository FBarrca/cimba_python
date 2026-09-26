import cimba as cb
from typing import TypedDict
from tutorial.tut_4_2 import Harbor


class HarborSettings(TypedDict):
    mean_wind: float
    reference_depth: float
    arrival_rate: float
    percent_large: float
    num_tugs: int
    num_berths_small: int
    unload_avg_small: float
    unload_avg_large: float


def test_more_large_berths_reduces_delay():
    settings: HarborSettings = {
        "mean_wind": 5, "reference_depth": 15, "arrival_rate": 0.5,
        "percent_large": 1, "num_tugs": 6, "num_berths_small": 1,
        "unload_avg_small": 2, "unload_avg_large": 4,
    }
    one = Harbor(**settings, num_berths_large=1)
    two = Harbor(**settings, num_berths_large=2)
    first = cb.Experiment(one, window=cb.Window(duration=72), seed=42).run()
    second = cb.Experiment(two, window=cb.Window(duration=72), seed=42).run()
    assert not first.failed.any() and not second.failed.any()
    assert second[two].avg_time_large[0, 0] < first[one].avg_time_large[0, 0]
