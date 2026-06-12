.. _installation:

Installation guide
==================

You will need Python 3.13, a C compiler like gcc or clang, the NASM assembler,
and a development toolchain of git, Meson, and Ninja. The easiest way to get the
Python side of that toolchain is to use ``uv``; it creates the virtual
environment, installs the Python build dependencies, and drives the editable
build.

The Python package wraps the native Cimba library vendored under
``subprojects/cimba``. The C library is built as a static archive and embedded
into the Python extension, so a wheel does not need a system-wide Cimba
installation at runtime.

Once the build chain is installed, you need to obtain the Python wrapper source.
The repository contains Cimba as a git submodule, so initialize submodules before
building.

Linux
-----

On Linux, open a terminal window, change directory to where you want to have the
source code tree, and enter these commands:

.. code-block:: bash

    git clone <repo-url> cimba_python
    cd cimba_python
    git submodule update --init --recursive
    uv sync

On Ubuntu or WSL, the system packages needed for the native build are usually:

.. code-block:: bash

    sudo apt install build-essential nasm libhdf5-dev

No manual virtual environment activation is needed. Use ``uv run`` for commands
inside the project environment.

Windows
-------

As always, things are more complicated on Windows. So far, the bundled Cimba
native library follows the upstream MinGW-W64 build chain with the gcc or clang
compiler. MSVC is not yet supported by the upstream C library.

To use the MinGW-W64 build chain, first make sure you have ``gcc`` installed and
in your PATH by typing ``gcc --version`` in a command shell. If it does not
respond, install MinGW-W64 before continuing. You will also need NASM and ``uv``
available in PATH.

From a command shell, change directory to where you want to have the Python
wrapper source hierarchy and issue the same project commands:

.. code-block:: batch

   git clone <repo-url> cimba_python
   cd cimba_python
   git submodule update --init --recursive
   uv sync

If Windows Security ransomware protection is enabled, you may have to allow
access for the compiler, assembler, Meson, and Ninja to your build folders. You
may also encounter issues with incompatible DLLs already installed by other
applications. Windows loads the first matching DLL in your ``PATH``, which may be
older and incompatible with newly compiled source code. This may be solved by
reordering the items in your ``PATH`` or updating the other applications.

Verifying your installation
---------------------------

After installation, verify that Python can import the wrapper and that it links
to the native Cimba library:

.. code-block:: bash

    uv run python -c "import cimba; print(cimba.native_version())"

If all goes well, this program should produce output similar to::

    3.0.0-beta

You now have a working Cimba Python installation.

For a more comprehensive test, type:

.. code-block:: bash

    uv run pytest

This will run the Python smoke tests from the ``tests`` directory. The tests
build and execute small ``cimba.sim`` models, exercise the random distributions,
process signals, resources, queues, stores, priority queues, conditions, and the
parallel experiment wrapper.

Running the tutorials
---------------------

The Python tutorial sources mirror the upstream C tutorial names in the
``tutorial`` directory. For example:

.. code-block:: bash

    uv run python tutorial/hello.py
    uv run python tutorial/tut_1_7.py -n 10 -d 1000000 -w 1000 -t

The larger tutorial ports reuse the maintained examples for resource
preemption, amusement-park queues, and harbor conditions.

Building a wheel
----------------

Build the normal release-speed wheel with:

.. code-block:: bash

    uv build --wheel

Release wheels compile out Cimba's native INFO trace for speed. To build a
tutorial/debugging wheel with those detailed native logs enabled, pass the
``cimba_debug_logs`` Meson option. The command below uses the PyPA ``build``
frontend; install it first if it is not already available in your environment:

.. code-block:: bash

    python -m pip install build
    python -m build --wheel \
      --config-setting=setup-args=-Dbuildtype=debugoptimized \
      --config-setting=setup-args=-Dcimba_debug_logs=true

That build keeps the bundled Cimba library optimized, but avoids the upstream
``-DNLOGINFO`` release flag so ``cimba.LOGGER_INFO`` can show internal process,
queue, timer, and dispatcher logs.

The wheel statically embeds Cimba. You can verify it in a throwaway environment:

.. code-block:: bash

    uv run --no-project --isolated \
      --with dist/cimba-*.whl \
      python -c "import cimba; print(cimba.native_version())"

Troubleshooting
---------------

If an editable install fails to import after deleting the build directory, force
``uv`` to rebuild the local package:

.. code-block:: bash

    uv sync --reinstall-package cimba

Use the same command if a plain ``uv sync`` does not pick up changes to
``meson.build`` or ``pyproject.toml``.
