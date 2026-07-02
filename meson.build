# Build for the Cimba Python wrapper: consume the upstream Cimba C library
# (vendored as a git submodule under subprojects/cimba), generate the cffi
# glue module, and statically link libcimba into it.
project(
    'cimba',
    'c',
    'cython',
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

py.install_sources(
    'src/cimba/__init__.py',
    'src/cimba/_cimba.py',
    'src/cimba/sim.py',
    'src/cimba/_bindings.py',
    'src/cimba/_graph.py',
    'src/cimba/_intrinsics.py',
    'src/cimba/_model.py',
    'src/cimba/cimba.py',
    'src/cimba/cmb_buffer.py',
    'src/cimba/cmb_condition.py',
    'src/cimba/cmb_datasummary.py',
    'src/cimba/cmb_dataset.py',
    'src/cimba/cmb_event.py',
    'src/cimba/cmb_logger.py',
    'src/cimba/cmb_objectqueue.py',
    'src/cimba/cmb_priorityqueue.py',
    'src/cimba/cmb_process.py',
    'src/cimba/cmb_random.py',
    'src/cimba/cmb_resource.py',
    'src/cimba/cmb_resourcepool.py',
    'src/cimba/cmb_timeseries.py',
    'src/cimba/cmb_wtdsummary.py',
    'src/cimba/reporting.py',
    'src/cimba/py.typed',
    'src/cimba/__init__.pyi',
    'src/cimba/_cimba.pyi',
    'src/cimba/_types.pyi',
    'src/cimba/cimba.pyi',
    'src/cimba/cmb_buffer.pyi',
    'src/cimba/cmb_condition.pyi',
    'src/cimba/cmb_datasummary.pyi',
    'src/cimba/cmb_dataset.pyi',
    'src/cimba/cmb_event.pyi',
    'src/cimba/cmb_logger.pyi',
    'src/cimba/cmb_objectqueue.pyi',
    'src/cimba/cmb_priorityqueue.pyi',
    'src/cimba/cmb_process.pyi',
    'src/cimba/cmb_random.pyi',
    'src/cimba/cmb_resource.pyi',
    'src/cimba/cmb_resourcepool.pyi',
    'src/cimba/cmb_timeseries.pyi',
    'src/cimba/cmb_wtdsummary.pyi',
    subdir: 'cimba',
)

cimba_native_ext = py.extension_module(
    '_cimba_native',
    'src/cimba/_cimba_native.pyx',
    # nbshim.c re-exports the static-inline cimba functions that sim.py binds
    # as Numba external symbols (cpy_*).
    'src/cimba/native/nbshim.c',
    include_directories: [inc_api, inc_int, 'src/cimba/native'],
    dependencies: [m_dep, thread_dep],
    link_args: ['-Wl,--wrap=cmi_coroutine_context_switch'],
    link_whole: cimba_lib,
    install: true,
    subdir: 'cimba',
)

# Upstream adds its tutorials/benchmarks/tests to the default `all` target (one
# links system HDF5). meson-python compiles only this alias instead -- see
# tool.meson-python.args.compile in pyproject.toml.
alias_target('cimba_ext', cimba_native_ext)

if get_option('enable_docs')
    subdir('docs')
endif
