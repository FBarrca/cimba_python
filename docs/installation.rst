.. _installation:

Installation
============

Python 3.13 or newer is required. Install a wheel with ``pip install cimba``
or ``uv add cimba``. The Linux x86_64, Windows AMD64, and macOS arm64 wheels
embed the native engine.

For a source checkout, initialize the unchanged engine submodule, install a C
compiler and NASM, then run:

.. code-block:: bash

   git submodule update --init --recursive
   uv sync --locked
   uv run python -c "import cimba; print(cimba.engine_version())"
   uv run pytest

On Apple Silicon, install Xcode Command Line Tools for a source build. On
Windows use clang-cl and NASM. On Linux install the platform compiler and
NASM. The test suite includes native execution and every standalone tutorial.
