from tutorial import hello


def test_hello_reports_cimba_version():
    assert hello.message().startswith("Hello world, I am Cimba 3.")
