"""Helpers for tutorial wrappers around the larger Python demo ports."""

from __future__ import annotations

import importlib
import pathlib
import sys


def run(module_name: str) -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module(module_name)
    module.main()

