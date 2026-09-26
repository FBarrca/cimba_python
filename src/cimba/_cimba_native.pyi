"""Private native runtime (libcimba + nbshim.c); it exports no Python
callables and is loaded by path for CFFI and Numba symbol resolution."""
