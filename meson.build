# Build for the Cimba Python wrapper: consume the upstream Cimba C library
# (vendored as a git submodule under subprojects/cimba), generate the cffi
# glue module, and statically link libcimba into it.
project(
    'cimba',
    'c',
    version: '0.1.0',
    meson_version: '>=1.1.0',
    # Fast wheels by default. Use -Dcimba_debug_logs=true to build the bundled
    # Cimba library with native INFO logging for tutorials/debugging.
    default_options: ['buildtype=release'],
)

py = import('python').find_installation(pure: false)
cc = meson.get_compiler('c')

if get_option('cimba_debug_logs') and get_option('buildtype') == 'release'
    error(
        'cimba_debug_logs requires -Dbuildtype=debugoptimized because '
        + 'upstream Cimba release builds compile INFO logs out with -DNLOGINFO'
    )
endif

cimba_proj = subproject(
    'cimba',
    default_options: [
        'default_library=static',
        'enable_docs=false',
        'warning_level=1',
    ],
)
cimba_lib = cimba_proj.get_variable('cimba_lib')
# inc_api alone is not enough: upstream's public cmb_process.h includes the
# internal cmi_coroutine.h, which lives in src/ (inc_int).
inc_api = cimba_proj.get_variable('inc_api')
inc_int = cimba_proj.get_variable('inc_int')

m_dep = cc.find_library('m', required: true)
thread_dep = dependency('threads')

# cffi codegen at build time: _cffi_build.py emits _cimba.c.
cffi_gen = custom_target(
    input: files('src/cimba/_cffi_build.py'),
    output: '_cimba.c',
    command: [py, '@INPUT@', '@OUTDIR@'],
)

py.install_sources(
    'src/cimba/__init__.py',
    'src/cimba/sim.py',
    'src/cimba/_bindings.py',
    'src/cimba/_graph.py',
    'src/cimba/_intrinsics.py',
    'src/cimba/_model.py',
    'src/cimba/py.typed',
    'src/cimba/_cimba.pyi',
    subdir: 'cimba',
)

cimba_ext = py.extension_module(
    '_cimba',
    cffi_gen,
    # nbshim.c re-exports the static-inline cimba functions that sim.py binds
    # as Numba external symbols (cpy_*).
    'src/cimba/native/nbshim.c',
    include_directories: [inc_api, inc_int, 'src/cimba/native'],
    dependencies: [m_dep, thread_dep],
    link_whole: cimba_lib,
    install: true,
    subdir: 'cimba',
)

# Upstream adds its tutorials/benchmarks/tests to the default `all` target (one
# links system HDF5). meson-python compiles only this alias instead -- see
# tool.meson-python.args.compile in pyproject.toml.
alias_target('cimba_ext', cimba_ext)

if get_option('enable_docs')
    subdir('docs')
endif
