from __future__ import division

import numpy as np

# cimport the Cython declarations for numpy
cimport numpy as np

np.import_array()

DTYPE_FLOAT = np.float32

DTYPE_INT = np.int32

ctypedef np.float32_t DTYPE_FLOAT_t

ctypedef np.int32_t DTYPE_INT_t

def find_host_using_global_search(DTYPE_FLOAT_t x, DTYPE_FLOAT_t y,
        np.ndarray[DTYPE_FLOAT_t, ndim=1] x_nodes,
        np.ndarray[DTYPE_FLOAT_t, ndim=1] y_nodes,
        np.ndarray[DTYPE_INT_t, ndim=2] nv):
    assert x_nodes.dtype == DTYPE_FLOAT and y_nodes.dtype == DTYPE_FLOAT and nv.dtype == DTYPE_INT

    cdef int i # Loop counter
    cdef int n_elems = nv.shape[1] # No. of elements
    cdef np.ndarray[DTYPE_INT_t, ndim=1] vertices = np.empty(3, dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] x_tri = np.empty(3, dtype=DTYPE_FLOAT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] y_tri = np.empty(3, dtype=DTYPE_FLOAT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] phi = np.empty(3, dtype=DTYPE_FLOAT) 

    for i in range(n_elems):
        vertices = nv[:,i].squeeze()
        x_tri[:] = x_nodes[vertices]
        y_tri[:] = y_nodes[vertices]

        # Transform to natural coordinates
        get_barycentric_coords(x, y, x_tri, y_tri, phi)

        # Check to see if the particle is in the current element
        if phi.min() >= 0.0:
            return i
    raise ValueError('Particle not found.')

# create the wrapper code, with numpy type annotations
def get_barycentric_coords(DTYPE_FLOAT_t x, DTYPE_FLOAT_t y,
        np.ndarray[DTYPE_FLOAT_t, ndim=1] x_tri,
        np.ndarray[DTYPE_FLOAT_t, ndim=1] y_tri,
        np.ndarray[DTYPE_FLOAT_t, ndim=1] phi):
    assert x_tri.dtype == DTYPE_FLOAT and y_tri.dtype == DTYPE_FLOAT and phi.dtype == DTYPE_FLOAT

    cdef DTYPE_FLOAT_t a11, a12, a21, a22, det

    # Array elements
    a11 = y_tri[2] - y_tri[0]
    a12 = x_tri[0] - x_tri[2]
    a21 = y_tri[0] - y_tri[1]
    a22 = x_tri[1] - x_tri[0]
    
    # Determinant
    det = a11 * a22 - a12 * a21
    
    # Transformation to barycentric coordinates
    phi[0] = (a11*(x - x_tri[0]) + a12*(y - y_tri[0]))/det
    phi[1] = (a21*(x - x_tri[0]) + a22*(y - y_tri[0]))/det
    phi[2] = 1.0 - phi[0] - phi[1]

