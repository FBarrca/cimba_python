from tutorial import tut_5_1


def test_tut_5_1_reports_missing_gpu_hooks(capsys):
    tut_5_1.main()

    assert "not exposed" in capsys.readouterr().out
