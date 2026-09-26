/* Load the embedded Cimba runtime for CFFI and Numba symbol resolution.
 * Simulation operations are exposed through cimba.sim; this private module
 * deliberately has no Python-callable native operations.
 */
#include <Python.h>

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_cimba_native",
    .m_doc = "Private native runtime for compiled Cimba models.",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__cimba_native(void)
{
    return PyModule_Create(&module);
}
