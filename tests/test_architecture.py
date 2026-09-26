"""Package boundaries and public/native API contracts."""

import ast
from pathlib import Path

from cimba.engine.library import load


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "cimba"


def test_package_layers_and_module_names():
    pure = {"inputs", "modeling", "schema", "layout", "results", "analysis"}
    forbidden_pure = {"numba", "llvmlite", "ctypes"}
    forbidden_compiler = {"cimba.inputs", "cimba.experiments",
                          "cimba.results"}
    for path in PACKAGE.rglob("*.py"):
        assert path.name == "__init__.py" or not path.name.startswith("_"), path
        package = path.relative_to(PACKAGE).parts[0]
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        if package in pure:
            assert not any(name.split(".")[0] in forbidden_pure
                           for name in imports), path
        if package == "compiler":
            if path.name != "numba_compat.py":
                assert not any(name == "numba.core" or
                               name.startswith("numba.core.")
                               for name in imports), path
            assert not any(name == edge or name.startswith(edge + ".")
                           for name in imports
                           for edge in forbidden_compiler), path
            assert not any(name.endswith(".Assembly")
                           for name in imports), path


def test_windows_export_list_names_exist_in_native_library():
    exports = (PACKAGE / "native" / "cimba.def").read_text().splitlines()
    assert exports[0] == "EXPORTS"
    library = load()
    for name in exports[1:]:
        if name.strip():
            assert hasattr(library, name.strip()), name
