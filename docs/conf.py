from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "cimba"
with (ROOT / "pyproject.toml").open("rb") as f:
    release = version = tomllib.load(f)["project"]["version"]

copyright = "Francisco Barragán Castro 2025-26"

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
