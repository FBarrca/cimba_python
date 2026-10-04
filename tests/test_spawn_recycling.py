"""Our binding retires finished spawn stacks without changing the C engine."""

import os
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != 'linux' or shutil.which('cc') is None,
                    reason='counts guard-page allocations via Linux interposition')
@pytest.mark.parametrize('release', [False, True])
def test_finished_spawn_stacks_are_recycled(tmp_path, release):
    source = tmp_path / 'count_guards.c'
    library = tmp_path / 'count_guards.so'
    source.write_text('''
        #define _GNU_SOURCE
        #include <dlfcn.h>
        #include <stdatomic.h>
        #include <sys/mman.h>
        static atomic_int guards;
        int guard_count(void) { return atomic_load(&guards); }
        int mprotect(void *address, size_t length, int protection) {
            if (protection == PROT_NONE) atomic_fetch_add(&guards, 1);
            int (*real)(void *, size_t, int) = dlsym(RTLD_NEXT, "mprotect");
            return real(address, length, protection);
        }
    ''')
    subprocess.run(['cc', '-shared', '-fPIC', str(source), '-o', str(library), '-ldl'], check=True)
    script = tmp_path / 'spawn_probe.py'
    script.write_text('''
import ctypes
import os
import cimba as cb
class Child(cb.Model):
    @cb.process
    def finish(self):
        cb.hold(.1)
''' + ('        cb.release(self)\n' if release else '') + '''
class Parent(cb.Model):
    @cb.process
    def spawn(self):
        for _ in range(3000):
            cb.spawn(Child)
            cb.hold(1)
library = ctypes.CDLL(os.environ['CIMBA_TEST_GUARD_LIBRARY'])
before = library.guard_count()
result = cb.Experiment(Parent()).run(workers=1)
after = library.guard_count()
assert not result.failed.any()
# Previously each child allocated a new guarded stack, about 3000 calls.
assert after - before < 20, (before, after)
''')
    env = os.environ.copy()
    env['LD_PRELOAD'] = str(library)
    env['CIMBA_TEST_GUARD_LIBRARY'] = str(library)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
