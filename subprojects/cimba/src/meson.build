# src/meson.build

# Platform independent library source code
sources = files('cimba.c',
                'cmb_assert.c',
                'cmb_buffer.c',
                'cmb_condition.c',
                'cmb_dataset.c',
                'cmb_datasummary.c',
                'cmb_event.c',
                'cmb_logger.c',
                'cmb_objectqueue.c',
                'cmb_priorityqueue.c',
                'cmb_process.c',
                'cmb_random.c',
                'cmb_resource.c',
                'cmb_resourceguard.c',
                'cmb_resourcepool.c',
                'cmb_timeseries.c',
                'cmb_wtdsummary.c',
                'cmi_coroutine.c',
                'cmi_hashheap.c',
                'cmi_holdable.c',
                'cmi_mempool.c',
                'cmi_resourcebase.c',
                'cmi_thread.c'
)

inc_int = include_directories('.')

# Internal headers - installed to /usr/local/include
install_headers(
    'cmi_config.h',
    'cmi_coroutine.h',
    'cmi_dataset.h',
    'cmi_hashheap.h',
    'cmi_holdable.h',
    'cmi_slist.h',
    'cmi_mempool.h',
    'cmi_memutils.h',
    'cmi_process.h',
    'cmi_resourcebase.h',
    'cmi_thread.h'
)

# Platform and operating system dependent source code
subdir('port')

cimba_lib = library('cimba',
                    sources,
                    include_directories : [inc_api, inc_int],
                    dependencies : [project_deps, codegen_deps],
                    install : true,
                    version : meson.project_version(),
                    soversion : '1'
)

my_lib_dep = declare_dependency(
                    link_with : cimba_lib,
                    include_directories : [inc_api, inc_int]
)
