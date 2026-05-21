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
        'src/cimba/py.typed',
        'src/cimba/_cimba.pyi',  # type stub so editors get hover/autocomplete
    ),
    subdir: 'cimba',
)

# The Cython extension. Meson runs cython -> cc -> link automatically; the
# dependency provides cimba.h's include dir and the static archive to link.
cimba_ext = py.extension_module(
    '_cimba',
    'src/cimba/_cimba.pyx',
    dependencies: [cimba_dep],
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
