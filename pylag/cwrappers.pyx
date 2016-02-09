"""
For speed, several of PyLag's cython functions take carrays as arguments. Cython
does not appear to produce python wrappers for these functions, so here we
manually implement a set of wrappers that do this.
"""

include "constants.pxi"

import numpy as np

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
cimport pylag.interpolation as interp
from data_reader cimport DataReader

cpdef get_barycentric_coords(x, y, x_tri, y_tri):
    cdef DTYPE_FLOAT_t x_tri_c[N_VERTICES]
    cdef DTYPE_FLOAT_t y_tri_c[N_VERTICES]
    cdef DTYPE_FLOAT_t phi_c[N_VERTICES]
    cdef DTYPE_INT_t i
    
    if x_tri.shape[0] != N_VERTICES or y_tri.shape[0] != N_VERTICES:
        raise ValueError('1D array must be have a length of {}.'.format(N_VERTICES))
    
    for i in xrange(N_VERTICES):
        x_tri_c[i] = x_tri[i]
        y_tri_c[i] = y_tri[i]
    
    interp.get_barycentric_coords(x, y, x_tri_c, y_tri_c, phi_c)
    
    # Generate and pass back an array type python can understand
    phi_out = np.empty(N_VERTICES, dtype='float32')
    for i in xrange(N_VERTICES):
        phi_out[i] = phi_c[i]
    return phi_out

cpdef interpolate_within_element(var, phi):
    cdef DTYPE_FLOAT_t var_c[N_VERTICES]
    cdef DTYPE_FLOAT_t phi_c[N_VERTICES]
    cdef DTYPE_INT_t i
    
    if var.shape[0] != N_VERTICES or phi.shape[0] != N_VERTICES:
        raise ValueError('1D array must be have a length of {}.'.format(N_VERTICES))
    
    for i in xrange(N_VERTICES):
        var_c[i] = var[i]
        phi_c[i] = phi[i]
    
    return interp.interpolate_within_element(var_c, phi_c)

cpdef get_velocity(DataReader data_reader, t, x, y, z, host):
    cdef DTYPE_FLOAT_t vel_c[N_VERTICES]
    cdef DTYPE_INT_t i
    
    data_reader.get_velocity(t, x, y, z, host, vel_c)
    
    # Generate and pass back an array type python can understand
    vel_out = np.empty(N_VERTICES, dtype='float32')
    for i in xrange(N_VERTICES):
        vel_out[i] = vel_c[i]
    return vel_out