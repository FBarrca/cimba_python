"""Generate the cffi C glue for ``cimba._cimba_cffi``.

Meson invokes this script at build time; it emits ``_cimba.c`` into the
given output directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cffi import FFI

HERE = Path(__file__).resolve().parent

ffibuilder = FFI()

ffibuilder.cdef("""
    const char *cimba_version(void);
    void cimba_run_experiment(void *your_experiment_array,
                              uint64_t num_trials,
                              size_t trial_struct_size,
                              void (*your_trial_func)(void *));
    uint64_t cmb_random_hwseed(void);
    uint64_t cpy_process_sizeof(void);
    void cpy_logger_flags_on(uint32_t flags);
    void cpy_logger_flags_off(uint32_t flags);
    void cmb_logger_flags_off(uint32_t flags);
""")

ffibuilder.set_source(
    "cimba._cimba_cffi",
    '#include "cimba.h"\n#include "nbshim.h"',
)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <output_dir>")
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    ffibuilder.emit_c_code(str(out_dir / "_cimba_cffi.c"))
