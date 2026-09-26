# Build the Python-independent runtime against the unchanged Cimba submodule.
project(
    'cimba',
    'c',
    version: '0.7.0',
    meson_version: '>=1.1.0',
    # Fast wheels by default. Use -Dbuildtype=debugoptimized for native INFO logs.
    default_options: ['buildtype=release'],
)

py = import('python').find_installation(pure: false)
cc = meson.get_compiler('c')

cimba_default_options = [
    'default_library=static',
    'enable_docs=false',
    'warning_level=1',
    'external_context_switch_wrapper=false',
    'enable_extras=false',
]

cimba_proj = subproject('cimba', default_options: cimba_default_options)
cimba_lib = cimba_proj.get_variable('cimba_lib')
# inc_api alone is not enough: upstream's public cmb_process.h includes the
# internal cmi_coroutine.h, which lives in src/ (inc_int).
inc_api = cimba_proj.get_variable('inc_api')
inc_int = cimba_proj.get_variable('inc_int')

m_dep = cc.find_library('m', required: false)
thread_dep = dependency('threads')
native_dependencies = [thread_dep]
if m_dep.found()
    native_dependencies += m_dep
endif

# Numba resolves these functions by name through LLVM. Unlike ELF/Mach-O,
# PE/COFF does not expose symbols pulled from the static library unless the
# extension DLL exports them explicitly. Meson ignores this Microsoft module
# definition file when using non-Windows linkers.
windows_module_defs = files('src/cimba/native/cimba.def')[0]

py.install_sources(
    'src/cimba/__init__.py',
    'src/cimba/sim.py',
    'src/cimba/py.typed',
    subdir: 'cimba',
)

py.install_sources(
    'src/cimba/random/__init__.py',
    'src/cimba/random/__init__.pyi',
    'src/cimba/random/compiled.py',
    subdir: 'cimba/random',
)

py.install_sources(
    'src/cimba/bootstrap/__init__.py',
    subdir: 'cimba/bootstrap',
)

py.install_sources(
    'src/cimba/inputs/__init__.py',
    'src/cimba/inputs/sources.py',
    'src/cimba/inputs/dist.py',
    'src/cimba/inputs/bootstrap.py',
    'src/cimba/inputs/decompose.py',
    'src/cimba/inputs/fitted.py',
    'src/cimba/inputs/fit.py',
    subdir: 'cimba/inputs',
)

py.install_sources('src/cimba/modeling/__init__.py', subdir: 'cimba/modeling')
py.install_sources('src/cimba/schema/__init__.py', subdir: 'cimba/schema')
py.install_sources('src/cimba/layout/__init__.py', subdir: 'cimba/layout')
py.install_sources('src/cimba/engine/__init__.py', 'src/cimba/engine/abi.py',
                   subdir: 'cimba/engine')
py.install_sources('src/cimba/experiments/__init__.py',
                   'src/cimba/experiments/design.py',
                   'src/cimba/experiments/run.py',
                   subdir: 'cimba/experiments')
py.install_sources('src/cimba/engine/library.py', 'src/cimba/engine/symbols.py',
                   'src/cimba/engine/runtime.py',
                   'src/cimba/engine/random_symbols.py',
                   subdir: 'cimba/engine')
py.install_sources('src/cimba/compiler/__init__.py',
                   'src/cimba/compiler/handles.py',
                   'src/cimba/compiler/classes.py',
                   'src/cimba/compiler/numba_compat.py',
                   'src/cimba/compiler/views.py',
                   'src/cimba/compiler/spawning.py', subdir: 'cimba/compiler')
py.install_sources('src/cimba/results/__init__.py', subdir: 'cimba/results')
py.install_sources('src/cimba/analysis/__init__.py', subdir: 'cimba/analysis')
py.install_sources('src/cimba/diagrams/__init__.py',
                   'src/cimba/diagrams/graph.py',
                   'src/cimba/diagrams/processes.py',
                   subdir: 'cimba/diagrams')

# The only native artifact has no Python headers or interpreter linkage.
cimba_native_lib = shared_library(
    'cimba_py',
    'src/cimba/native/nbshim.c',
    'src/cimba/native/inputs.c',
    'src/cimba/native/trial.c',
    'src/cimba/native/verbs.c',
    include_directories: [inc_api, inc_int, 'src/cimba/native'],
    dependencies: native_dependencies,
    link_whole: cimba_lib,
    vs_module_defs: windows_module_defs,
    install: true,
    install_dir: py.get_install_dir() / 'cimba',
)

# Stable target for meson-python's editable rebuilds.
alias_target('cimba_ext', [cimba_native_lib])
