import os
import re
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "cimba"
with open(os.path.join(os.path.dirname(__file__), "../pyproject.toml"), "r") as f:
    pyproject = f.read()
version_match = re.search(r"version\s*=\s*\"([^\"]+)\"", pyproject)
if version_match:
    release = version_match.group(1)
    version = release
else:
    release = "0.1.0"
    version = release

copyright = "Asbjorn M. Bonvik 2025-26"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_mock_imports = ["cimba._cimba", "numba", "llvmlite"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
}
html_static_path = ["./static"]
html_css_files = ["custom.css"]

primary_domain = "py"
highlight_language = "python"

# External links to the upstream Cimba C documentation on Read the Docs.
rst_epilog = """
.. _Cimba C documentation: https://cimba.readthedocs.io/en/latest/
.. _Cimba C tutorial: https://cimba.readthedocs.io/en/latest/tutorial.html
.. _Cimba C background: https://cimba.readthedocs.io/en/latest/background.html
.. _Cimba C API reference: https://cimba.readthedocs.io/en/latest/api/library_root.html
.. _Cimba C installation guide: https://cimba.readthedocs.io/en/latest/installation.html
"""
