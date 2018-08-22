""" PyLag cwrappers

For speed, several of PyLag's Cython functions take c arrays or pointers as 
arguments. Cython does not appear to produce python wrappers for functions 
that have arguments with these types. The wrappers provided here are intended
to allow clients to call these functions in isolation.
"""

include "constants.pxi"

import numpy as np

# PyLag data types
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from libcpp.vector cimport vector

# PyLag cimports
cimport pylag.math as math
cimport pylag.interpolation as interp
from pylag.data_reader cimport DataReader
from pylag.particle cimport ParticleSmartPtr

def det_wrapper(a, b):
    cdef DTYPE_FLOAT_t a_c[2]
    cdef DTYPE_FLOAT_t b_c[2]
    a_c[:] = a[:]
    b_c[:] = b[:]
    return math.det(a_c, b_c)

def inner_product_wrapper(a, b):
    cdef DTYPE_FLOAT_t a_c[2]
    cdef DTYPE_FLOAT_t b_c[2]
    a_c[:] = a[:]
    b_c[:] = b[:]
    return math.inner_product(a_c, b_c)

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
    phi_out = np.empty(N_VERTICES, dtype=DTYPE_FLOAT)
    for i in xrange(N_VERTICES):
        phi_out[i] = phi_c[i]
    return phi_out

cpdef get_barycentric_gradients(x_tri, y_tri):
    cdef DTYPE_FLOAT_t x_tri_c[N_VERTICES]
    cdef DTYPE_FLOAT_t y_tri_c[N_VERTICES]
    cdef DTYPE_FLOAT_t dphi_dx_c[N_VERTICES]
    cdef DTYPE_FLOAT_t dphi_dy_c[N_VERTICES]
    cdef DTYPE_INT_t i

    if x_tri.shape[0] != N_VERTICES or y_tri.shape[0] != N_VERTICES:
        raise ValueError('1D array must be have a length of {}.'.format(N_VERTICES))
    
    for i in xrange(N_VERTICES):
        x_tri_c[i] = x_tri[i]
        y_tri_c[i] = y_tri[i]
    
    interp.get_barycentric_gradients(x_tri_c, y_tri_c, dphi_dx_c, dphi_dy_c)
    
    # Generate and pass back an array type python can understand
    dphi_dx_out = np.empty(N_VERTICES, dtype=DTYPE_FLOAT)
    dphi_dy_out = np.empty(N_VERTICES, dtype=DTYPE_FLOAT)
    for i in xrange(N_VERTICES):
        dphi_dx_out[i] = dphi_dx_c[i]
        dphi_dy_out[i] = dphi_dy_c[i]
    return dphi_dx_out, dphi_dy_out

cpdef shepard_interpolation(x, y, xpts, ypts, vals):
    cdef vector[DTYPE_FLOAT_t] xpts_c, ypts_c, vals_c
    
    if xpts.shape[0] != N_NEIGH_ELEMS or ypts.shape[0] != N_NEIGH_ELEMS or vals.shape[0] != N_NEIGH_ELEMS:
        raise ValueError('1D arrays should be {} elements in length.'.format(N_NEIGH_ELEMS))

    for i in xrange(N_NEIGH_ELEMS):
        xpts_c.push_back(xpts[i])
        ypts_c.push_back(ypts[i])  
        vals_c.push_back(vals[i])
    
    return interp.shepard_interpolation(x, y, xpts_c, ypts_c, vals_c)

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

cpdef set_vertical_grid_vars(DataReader data_reader, time, xpos, ypos, zpos, 
        host):
    cdef ParticleSmartPtr particle
    
    particle = ParticleSmartPtr(xpos=xpos, ypos=ypos, zpos=zpos, host=host)

    data_reader.set_local_coordinates(particle.get_ptr())
    data_reader.set_vertical_grid_vars(time, particle.get_ptr())
    
    grid_vars = {}
    grid_vars['k_layer'] = particle.get_ptr().k_layer
    grid_vars['k_lower_layer'] = particle.get_ptr().k_lower_layer
    grid_vars['k_upper_layer'] = particle.get_ptr().k_upper_layer
    grid_vars['in_vertical_boundary_layer'] = particle.get_ptr().in_vertical_boundary_layer
    grid_vars['omega_interfaces'] = particle.get_ptr().omega_interfaces
    grid_vars['omega_layers'] = particle.get_ptr().omega_layers
    
    return grid_vars

cpdef get_velocity(DataReader data_reader, time, xpos, ypos, zpos, host, k_layer):
    cdef ParticleSmartPtr particle
    cdef DTYPE_FLOAT_t vel_c[N_VERTICES]

    particle = ParticleSmartPtr(xpos=xpos, ypos=ypos, zpos=zpos, host=host,
            k_layer=k_layer)

    data_reader.set_local_coordinates(particle.get_ptr())
    data_reader.set_vertical_grid_vars(time, particle.get_ptr())

    data_reader.get_velocity(time, particle.get_ptr(), vel_c)
    
    # Generate and pass back an array type python can understand
    vel_out = np.empty(N_VERTICES, dtype=DTYPE_FLOAT)
    for i in xrange(N_VERTICES):
        vel_out[i] = vel_c[i]
    return vel_out

cpdef get_horizontal_velocity(DataReader data_reader, time, xpos, ypos, zpos,
        host, k_layer):
    cdef ParticleSmartPtr particle
    cdef DTYPE_FLOAT_t vel_c[2]
    cdef DTYPE_INT_t i

    particle = ParticleSmartPtr(xpos=xpos, ypos=ypos, zpos=zpos, host=host,
            k_layer=k_layer)

    data_reader.set_local_coordinates(particle.get_ptr())
    data_reader.set_vertical_grid_vars(time, particle.get_ptr())

    data_reader.get_horizontal_velocity(time, particle.get_ptr(), vel_c)

    # Generate and pass back an array python can understand
    vel_out = np.empty(N_VERTICES, dtype=DTYPE_FLOAT)
    for i in xrange(2):
        vel_out[i] = vel_c[i]
    return vel_out

cpdef get_horizontal_eddy_viscosity_derivative(DataReader data_reader, time, xpos, ypos, zpos, host):
    cdef ParticleSmartPtr particle
    cdef DTYPE_FLOAT_t Ah_prime_c[2]
    cdef DTYPE_INT_t i

    particle = ParticleSmartPtr(xpos=xpos, ypos=ypos, zpos=zpos, host=host)

    data_reader.set_local_coordinates(particle.get_ptr())
    data_reader.set_vertical_grid_vars(time, particle.get_ptr())

    data_reader.get_horizontal_eddy_viscosity_derivative(time, particle.get_ptr(), Ah_prime_c)

    # Generate and pass back an array python can understand
    Ah_prime_out = np.empty(2, dtype=DTYPE_FLOAT)
    for i in xrange(2):
        Ah_prime_out[i] = Ah_prime_c[i]
    return Ah_prime_out

def get_intersection_point_wrapper(x1, x2, x3, x4, xi):
    cdef DTYPE_FLOAT_t x1_c[2]
    cdef DTYPE_FLOAT_t x2_c[2]
    cdef DTYPE_FLOAT_t x3_c[2]
    cdef DTYPE_FLOAT_t x4_c[2]
    cdef DTYPE_FLOAT_t xi_c[2]
            
    if x1.shape[0] != 2 or x2.shape[0] != 2 or x3.shape[0] != 2 or x4.shape[0] != 2:
        raise ValueError('1D arrays must contain two elements.')
    
    x1_c[:] = x1[:]
    x2_c[:] = x2[:]
    x3_c[:] = x3[:]
    x4_c[:] = x4[:]
    
    math.get_intersection_point(x1_c, x2_c, x3_c, x4_c, xi_c)
    for i in xrange(2):
        xi[i] = xi_c[i]
    return
