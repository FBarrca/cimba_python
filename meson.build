# Top-level build for the Cimba Python wrapper.
#
# Strategy: consume the upstream Cimba C library (vendored as a git submodule
# under subprojects/cimba) as a Meson subproject, then compile our Cython
# extension and statically link it against that library. Building Cimba as a
# static lib keeps the wheel self-contained (a single _cimba*.so, no external
# .so to ship or RPATH to fix up).
project(
    'cimba',
    'c', 'cython',
    version: '0.1.0',
    meson_version: '>=1.1.0',
    default_options: [
        'buildtype=release',
        'default_library=static',
    ],
)

py = import('python').find_installation(pure: false)

# Pull in the upstream C library. enable_docs=false skips the doxygen/sphinx
# subdir; building Cimba static lets us embed it into the extension module.
cimba_proj = subproject(
    'cimba',
    default_options: [
        'default_library=static',
        'enable_docs=false',
        'warning_level=1',
    ],
)
# Upstream declares this in src/meson.build (it does not override_dependency,
# so we fetch the variable directly). It carries link_with + include dirs.
cimba_dep = cimba_proj.get_variable('my_lib_dep')

# Pure-Python parts of the package.
py.install_sources(
    files(
        'src/cimba/__init__.py',
        'src/cimba/__init__.pyi',
        'src/cimba/py.typed',
        'src/cimba/cimba.py',
        'src/cimba/cimba.pyi',
        'src/cimba/cmb_buffer.py',
        'src/cimba/cmb_buffer.pyi',
        'src/cimba/cmb_condition.py',
        'src/cimba/cmb_condition.pyi',
        'src/cimba/cmb_dataset.py',
        'src/cimba/cmb_dataset.pyi',
        'src/cimba/cmb_datasummary.py',
        'src/cimba/cmb_datasummary.pyi',
        'src/cimba/cmb_event.py',
        'src/cimba/cmb_event.pyi',
        'src/cimba/cmb_logger.py',
        'src/cimba/cmb_logger.pyi',
        'src/cimba/cmb_objectqueue.py',
        'src/cimba/cmb_objectqueue.pyi',
        'src/cimba/cmb_priorityqueue.py',
        'src/cimba/cmb_priorityqueue.pyi',
        'src/cimba/cmb_process.py',
        'src/cimba/cmb_process.pyi',
        'src/cimba/cmb_random.py',
        'src/cimba/cmb_random.pyi',
        'src/cimba/cmb_resource.py',
        'src/cimba/cmb_resource.pyi',
        'src/cimba/cmb_resourcepool.py',
        'src/cimba/cmb_resourcepool.pyi',
        'src/cimba/cmb_timeseries.py',
        'src/cimba/cmb_timeseries.pyi',
        'src/cimba/cmb_wtdsummary.py',
        'src/cimba/cmb_wtdsummary.pyi',
        'src/cimba/_types.pyi',
        'src/cimba/_cimba.pyi',  # type stub so editors get hover/autocomplete
    ),
    subdir: 'cimba',
)

# The Cython extension. Meson runs cython -> cc -> link automatically; the
# dependency provides cimba.h's include dir and the static archive to link.
# -Wl,--wrap=cmi_coroutine_context_switch interposes the vendored coroutine
# context switch at link time (no submodule changes) so the binding can save and
# restore CPython per-thread frame/exception state across switches on Python
# 3.14+. See src/cimba/_cython/cmb_corostate.pxi.
cimba_ext = py.extension_module(
    '_cimba',
    'src/cimba/_cimba.pyx',
    dependencies: [cimba_dep],
    link_args: ['-Wl,--wrap=cmi_coroutine_context_switch'],
    install: true,
    subdir: 'cimba',
)

# Upstream adds its tutorials/benchmarks/tests to Meson's default `all` target
# (it doesn't guard them with meson.is_subproject()), and one tutorial links
# system HDF5. We don't ship any of those. This stable-named alias lets the
# build compile ONLY our extension and its real deps (libcimba + codegen) -- see
# `tool.meson-python.args.compile` in pyproject.toml -- so the examples (and the
# HDF5 lookup) are never built.
alias_target('cimba_ext', cimba_ext)

if get_option('enable_docs')
    subdir('docs')
endif
