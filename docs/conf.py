import os
import re
import tomllib


project = "cimba-python"

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
with open(os.path.join(root, "pyproject.toml"), "rb") as f:
    pyproject = tomllib.load(f)

release = pyproject["project"]["version"]
version_match = re.match(r"^(\d+\.\d+)", release)
version = version_match.group(1) if version_match else release

copyright = "FBarrca 2026"

extensions = []

html_theme = "sphinx_rtd_theme"
html_logo = "../subprojects/cimba/images/logo_small.png"
html_theme_options = {
    "logo_only": False,
}
html_static_path = ["./static"]
html_css_files = ["custom.css"]

primary_domain = "py"
highlight_language = "python"

rst_epilog = """
.. _Cimba C documentation: https://cimba.readthedocs.io/en/latest/
.. _Cimba C tutorial: https://cimba.readthedocs.io/en/latest/tutorial.html
.. _Cimba C background: https://cimba.readthedocs.io/en/latest/background.html
.. _Cimba C API reference: https://cimba.readthedocs.io/en/latest/api/library_root.html
.. _Cimba C installation guide: https://cimba.readthedocs.io/en/latest/installation.html
"""
