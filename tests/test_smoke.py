"""Smoke tests: confirm the package built and links the native C library.

These do not test simulation behaviour (none is wrapped yet) — they verify the
build pipeline produced an importable extension that can call into Cimba.
"""

import cimba


def test_wrapper_version():
    assert cimba.__version__ == "0.1.0"


def test_native_version_is_linked():
    # Calling into the C library proves the extension is actually linked
    # against libcimba, not merely importable.
    v = cimba.native_version()
    assert isinstance(v, str)
    assert v, "native version string should not be empty"
    # Submodule is pinned to Cimba 3.x; keep this loose to survive bumps.
    assert v.startswith("3."), f"unexpected native version: {v!r}"
