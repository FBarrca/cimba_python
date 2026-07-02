from tutorial import hello


def test_hello_reports_cimba_version(capsys):
    hello.main()

    assert capsys.readouterr().out.startswith("Hello world, I am Cimba 3.")
