.. _installation:

Installation guide
==================

You will need Python 3.13, ``uv``, git, and the local build tools required by
the Python extension. The package builds and embeds its bundled simulation
engine, so you do not need to install a separate runtime library before using
Cimba Python.

Linux
-----

On Linux, clone the repository, initialize submodules, and let ``uv`` create the
project environment:

.. code-block:: bash

    git clone <repo-url> cimba_python
    cd cimba_python
    git submodule update --init --recursive
    uv sync

On Ubuntu or WSL, install the usual native build packages first:

.. code-block:: bash

    sudo apt install build-essential nasm libhdf5-dev

Use ``uv run`` for commands that should execute inside the project
environment.

Windows
-------

On Windows, install Python 3.13, ``uv``, git, NASM, and a supported compiler
toolchain. Then run the same project commands from a developer shell:

.. code-block:: batch

   git clone <repo-url> cimba_python
   cd cimba_python
   git submodule update --init --recursive
   uv sync

If Windows Security blocks build tools from writing into the project directory,
allow the compiler, assembler, and Python build tools. If imports fail because
another application provides incompatible runtime DLLs earlier on ``PATH``,
adjust ``PATH`` so the active compiler environment comes first.

Verifying your installation
---------------------------

Verify that Python can import the package:

.. code-block:: bash

    uv run python -c "import cimba; print(cimba.native_version())"

If all goes well, this prints a version such as::

    3.0.0-beta

Run the test suite with:

.. code-block:: bash

    uv run pytest

The tests build and execute small ``cimba.sim`` models covering imports,
logging, declarations, queues, resources, pools, stores, priority queues,
conditions, random draws, process signals, timers, dynamic processes, events,
and parallel experiments.
