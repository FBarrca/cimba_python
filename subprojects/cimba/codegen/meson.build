# codegen/meson.build

# Compile generator executables for the build machine. They run during the
# build even when the library itself is being cross-compiled.
build_cc = meson.get_compiler('c', native : true)
build_math_dep = build_cc.find_library('m', required : false)
build_codegen_deps = []
if build_math_dep.found()
    build_codegen_deps += build_math_dep
endif

genexp_exe = executable('calc_exponential',
                        files('calc_exponential.c','calc_utils.c'),
                        dependencies : build_codegen_deps,
                        native : true
)

gennor_exe = executable('calc_normal',
                        files('calc_normal.c', 'calc_utils.c'),
                        dependencies : build_codegen_deps,
                        native : true
)

# Define code generation step
exp_inc = custom_target('cmi_random_exp_zig.inc',
                        output : 'cmi_random_exp_zig.inc',
                        command : [genexp_exe],
                        capture : true
)

nor_inc = custom_target('cmi_random_nor_zig.inc',
                        output : 'cmi_random_nor_zig.inc',
                        command : [gennor_exe],
                        capture : true
)

# Expose these targets
codegen_deps = declare_dependency(sources : [exp_inc, nor_inc],
                                  include_directories : '.'
)
