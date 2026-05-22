from tutorial import tut_1_7


def test_tut_1_7_command_line_parameters_map_to_trial_fields():
    trial = tut_1_7.run_from_args(
        [
            "--arr-rate",
            "0.5",
            "--srv-rate",
            "1.0",
            "--warmup-time",
            "10.0",
            "--duration",
            "1000.0",
            "--seed",
            "17",
        ]
    )

    assert trial.arr_rate == 0.5
    assert trial.srv_rate == 1.0
    assert trial.avg_queue_length > 0.0
