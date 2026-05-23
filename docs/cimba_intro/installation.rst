Installation
============

Cimba Python vendors the Cimba C library and links it statically into the
``cimba._cimba`` extension module. You do not need to install the C library
separately to use the Python package.

Prerequisites
-------------

You need:

* Python 3.13 or newer
* git
* a C compiler such as gcc or clang
* NASM
* Meson, Ninja, Cython, and meson-python

The project is set up for ``uv``. On Ubuntu or WSL, the native tools are
typically installed with:

.. code-block:: bash

   sudo apt install build-essential nasm git

Fresh clone
-----------

.. code-block:: bash

   git clone <repo-url> cimba_python
   cd cimba_python
   git submodule update --init --recursive
   uv sync
   uv run python -c "import cimba; print(cimba.native_version())"
   uv run pytest

The first ``uv sync`` compiles the vendored C library and the Cython extension.
Later runs are incremental.

If an editable install stops importing after a build directory was removed,
force a package rebuild:

.. code-block:: bash

   uv sync --reinstall-package cimba

Build these docs
----------------

The docs use Sphinx and the Read the Docs theme, following the style of the
vendored C docs in ``subprojects/cimba/docs``.

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   sphinx-build -b html docs build/docs/html

You can also build them through Meson:

.. code-block:: bash

   meson setup build-docs -Denable_docs=true
   meson compile -C build-docs python-docs

The direct Sphinx command writes to ``build/docs/html``. The Meson target writes
to ``build-docs/docs/html``.

Native Cimba
------------

You do not need a native Cimba installation for Python imports. If you want to
build or use Cimba's C API directly, follow the `Cimba C installation guide`_.
