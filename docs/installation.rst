.. _installation:

Installation
============

From PyPI
---------

Cimba Python requires **Python 3.13 or newer**.

.. code-block:: bash

   pip install cimba
   # or
   uv add cimba

Pre-built wheels are published for:

.. list-table::
   :header-rows: 1

   * - Platform
     - Architecture
     - Notes
   * - Linux (manylinux)
     - x86_64
     -
   * - Windows
     - AMD64
     - built with clang-cl
   * - macOS
     - arm64 (Apple Silicon)
     -

Each wheel embeds the native Cimba engine and the Cimba Python runtime as one
C library that doesn't link against Python, so one wheel serves every supported
CPython version on that platform.

Runtime dependencies are installed automatically: NumPy, Numba (and
llvmlite), SciPy and statsmodels. The optional ``plot`` extra adds matplotlib
for the plotting examples:

.. code-block:: bash

   pip install "cimba[plot]"

Verify the installation:

.. code-block:: bash

   python -c "import cimba; print(cimba.__version__, cimba.engine_version())"

From a source checkout
----------------------

You need a C compiler and the `NASM <https://www.nasm.us/>`_ assembler (the
engine's coroutine context switch is written in assembly), plus
`uv <https://docs.astral.sh/uv/>`_.

.. code-block:: bash

   git clone https://github.com/FBarrca/cimba_python
   cd cimba_python
   git submodule update --init --recursive     # the Cimba C engine
   uv sync --locked                            # builds the native library
   uv run python -c "import cimba; print(cimba.engine_version())"
   uv run pytest

Platform notes:

* **Linux:** install ``gcc`` (or ``clang``) and ``nasm`` from your package
  manager, for example ``apt install build-essential nasm``.
* **macOS (Apple Silicon):** install the Xcode Command Line Tools
  (``xcode-select --install``) and ``brew install nasm``.
* **Windows:** use ``clang-cl`` with NASM on the ``PATH``.

The checkout is an editable install: after editing C sources, the library is
rebuilt on the next ``import cimba``.

Building this documentation
---------------------------

.. code-block:: bash

   uv pip install -r docs/requirements.txt
   uv run sphinx-build -W -b html docs build/docs/html

Every model in ``tutorial/`` is exercised by the test suite
(``tests/tutorial``), so the tutorial code shown in these pages is checked on
every change.
