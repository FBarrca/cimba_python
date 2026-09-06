# Native lifecycle sanitizer regression

`lifecycle.c` links the actual binding shims with the bundled Cimba library.
It runs 384 trials across repeated experiments with one and four workers.
Half abandon through `cmb_logger_error()` from a running process, while other
processes are suspended. The successful trials stop, terminate, and destroy
their processes and entities, then assert that the native memory registry is
empty. Every trial also exercises temporary dataset and weighted summaries.

Run from the repository root on Linux with GCC, Meson, Ninja, and NASM installed:

```sh
lifecycle_dir=$(mktemp -d /tmp/cimba-lifecycle-XXXXXX)
.venv/bin/meson setup "$lifecycle_dir/build" subprojects/cimba \
    --buildtype=debug -Db_sanitize=address -Ddefault_library=static \
    -Denable_docs=false -Denable_extras=false
.venv/bin/meson compile -C "$lifecycle_dir/build"
cc -std=c2x -g -O1 -fsanitize=address -fno-omit-frame-pointer \
    -D_POSIX_C_SOURCE=200809L \
    -I subprojects/cimba/include -I subprojects/cimba/src -I src/cimba/native \
    tests/native/lifecycle.c src/cimba/native/nbshim.c \
    "$lifecycle_dir/build/src/libcimba.a" -lm -pthread \
    -o "$lifecycle_dir/lifecycle"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    "$lifecycle_dir/lifecycle" > "$lifecycle_dir/run.log" 2>&1
```

A successful run exits zero without sanitizer reports. The 192 intentional
error messages in `run.log` are expected. No leak suppressions are used.
This checks native shim allocations and recovery; run `pytest` separately to
check Python integration. The standalone executable uses Cimba's native
context-switch wrapper, not the Python interpreter-state wrapper.
