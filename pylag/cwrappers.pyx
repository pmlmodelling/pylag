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

# PyLag python imports
from pylag.integrator import get_num_integrator
from pylag.random_walk import get_vertical_random_walk_model
from pylag.boundary_conditions import get_vert_boundary_condition_calculator

# PyLag cimports
cimport pylag.math as math
cimport pylag.interpolation as interp
from pylag.data_reader cimport DataReader
from pylag.particle cimport Particle
from pylag.delta cimport Delta, reset
from pylag.integrator cimport NumIntegrator
from pylag.random_walk cimport VerticalRandomWalk
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator

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

cpdef shepard_interpolation(x, y, xpts, ypts, vals):
    cdef DTYPE_FLOAT_t[N_NEIGH_ELEMS] xpts_c, ypts_c, vals_c
    
    if xpts.shape[0] != N_NEIGH_ELEMS or ypts.shape[0] != N_NEIGH_ELEMS or vals.shape[0] != N_NEIGH_ELEMS:
        raise ValueError('1D arrays should be {} elements in length.'.format(N_NEIGH_ELEMS))

    for i in xrange(N_NEIGH_ELEMS):
        xpts_c[i] = xpts[i]
        ypts_c[i] = ypts[i]   
        vals_c[i] = vals[i]
    
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

cpdef get_zmin(DataReader data_reader, time, xpos, ypos, host):
    cdef Particle particle
    particle.xpos = xpos
    particle.ypos = ypos
    particle.host_horizontal_elem = host

    data_reader.set_local_coordinates(&particle)
    return data_reader.get_zmin(time, &particle)

cpdef get_zmax(DataReader data_reader, time, xpos, ypos, host):
    cdef Particle particle
    particle.xpos = xpos
    particle.ypos = ypos
    particle.host_horizontal_elem = host

    data_reader.set_local_coordinates(&particle)
    return data_reader.get_zmax(time, &particle)

cpdef set_vertical_grid_vars(DataReader data_reader, t, x, y, z, host):
    cdef Particle particle

    particle.xpos = x
    particle.ypos = y
    particle.zpos = z
    particle.host_horizontal_elem = host

    data_reader.set_local_coordinates(&particle)
    data_reader.set_vertical_grid_vars(t, &particle)
    
    grid_vars = {}
    grid_vars['k_layer'] = particle.k_layer
    grid_vars['k_lower_layer'] = particle.k_lower_layer
    grid_vars['k_upper_layer'] = particle.k_upper_layer
    grid_vars['in_vertical_boundary_layer'] = particle.in_vertical_boundary_layer
    grid_vars['omega_interfaces'] = particle.omega_interfaces
    grid_vars['omega_layers'] = particle.omega_layers
    
    return grid_vars

cpdef get_velocity(DataReader data_reader, t, x, y, z, host, zlayer):
    cdef DTYPE_FLOAT_t vel_c[N_VERTICES]
    cdef Particle particle

    particle.xpos = x
    particle.ypos = y
    particle.zpos = z
    particle.host_horizontal_elem = host

    data_reader.set_local_coordinates(&particle)
    data_reader.set_vertical_grid_vars(t, &particle)
    data_reader.get_velocity(t, &particle, vel_c)
    
    # Generate and pass back an array type python can understand
    vel_out = np.empty(N_VERTICES, dtype=DTYPE_FLOAT)
    for i in xrange(N_VERTICES):
        vel_out[i] = vel_c[i]
    return vel_out

cpdef get_horizontal_velocity(DataReader data_reader, t, x, y, z, host, zlayer):
    cdef DTYPE_FLOAT_t vel_c[2]
    cdef Particle particle
    cdef DTYPE_INT_t i

    particle.xpos = x
    particle.ypos = y
    particle.zpos = z
    particle.host_horizontal_elem = host
    particle.k_layer = zlayer

    data_reader.set_local_coordinates(&particle)
    data_reader.set_vertical_grid_vars(t, &particle)
    data_reader.get_horizontal_velocity(t, &particle, vel_c)
    
    # Generate and pass back an array python can understand
    vel_out = np.empty(N_VERTICES, dtype=DTYPE_FLOAT)
    for i in xrange(2):
        vel_out[i] = vel_c[i]
    return vel_out

cpdef get_vertical_eddy_diffusivity(DataReader data_reader, time, xpos, ypos, zpos, host):
    cdef Particle particle
    particle.xpos = xpos
    particle.ypos = ypos
    particle.zpos = zpos
    particle.host_horizontal_elem = host

    data_reader.set_local_coordinates(&particle)
    data_reader.set_vertical_grid_vars(time, &particle)

    return data_reader.get_vertical_eddy_diffusivity(time, &particle)

cpdef get_vertical_eddy_diffusivity_derivative(DataReader data_reader, time, xpos, ypos, zpos, host):
    cdef Particle particle
    particle.xpos = xpos
    particle.ypos = ypos
    particle.zpos = zpos
    particle.host_horizontal_elem = host

    data_reader.set_local_coordinates(&particle)
    data_reader.set_vertical_grid_vars(time, &particle)

    return data_reader.get_vertical_eddy_diffusivity_derivative(time, &particle)

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

cdef class TestRK4Integrator:
    """ Test class for Fourth Order Runga Kutta numerical integration schemes
    
    Parameters:
    -----------
    config : SafeConfigParser
        Configuration object.
    """
    cdef NumIntegrator _num_integrator
    
    def __init__(self, config):
        
        self._num_integrator = get_num_integrator(config)
    
    def advect(self, data_reader, time, xpos, ypos, zpos):
        cdef Particle particle
        cdef Delta delta_X

        # Set these properties to default values
        particle.group_id = 0
        particle.host_horizontal_elem = 0
        particle.k_layer = 0
        particle.in_domain = True

        # Initialise remaining particle properties using the supplied arguments
        particle.xpos = xpos
        particle.ypos = ypos
        particle.zpos = zpos
        
        # Reset Delta object
        reset(&delta_X)
        
        # Advect the particle
        self._num_integrator.advect(time, &particle, data_reader, &delta_X)
        
        # Used Delta values to update the particle's position
        xpos_new = particle.xpos + delta_X.x
        ypos_new = particle.ypos + delta_X.y
        zpos_new = particle.zpos + delta_X.z

        # Return the updated position
        return xpos_new, ypos_new, zpos_new

cdef class TestVerticalRandomWalk:
    """ Test class for vertical random walk models.
    
    Parameters:
    -----------
    config : SafeConfigParser
        Configuration object.
    """
    cdef VerticalRandomWalk _vertical_random_walk
    cdef VertBoundaryConditionCalculator _vert_bc_calculator
    
    def __init__(self, config):

        self._vertical_random_walk = get_vertical_random_walk_model(config)

        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)
    
    def random_walk(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, zpos_arr, DTYPE_INT_t host):
        cdef Particle particle
        cdef Delta delta_X
        
        cdef DTYPE_FLOAT_t zpos_new, zmin, zmax
        
        cdef DTYPE_INT_t i, n_zpos

        # Set default particle properties
        particle.in_domain = True
        particle.group_id = 0

        # Use supplied args to set the host, x and y positions
        particle.host_horizontal_elem = host
        particle.xpos = xpos
        particle.ypos = ypos

        # Number of z positions
        n_zpos = len(zpos_arr)
        
        # Array in which to store updated z positions
        zpos_new_arr = np.empty(n_zpos, dtype=DTYPE_FLOAT)
        
        # Loop over the particle set
        for i in xrange(n_zpos):
            # Set zpos, local coordinates and variables that define the location
            # of the particle within the vertical grid
            particle.zpos = zpos_arr[i]
            data_reader.set_local_coordinates(&particle)
            data_reader.set_vertical_grid_vars(time, &particle)

            # Reset Delta object
            reset(&delta_X)

            # Apply the stochastic model
            self._vertical_random_walk.random_walk(time, &particle, data_reader, &delta_X)

            # Use Delta values to update the particle's position
            zpos_new = particle.zpos + delta_X.z
            
            # Apply boundary conditions
            zmin = data_reader.get_zmin(time, &particle)
            zmax = data_reader.get_zmax(time, &particle)
            if zpos_new < zmin or zpos_new > zmax:
                zpos_new = self._vert_bc_calculator.apply(zpos_new, zmin, zmax)
            
            # Set new z position
            zpos_new_arr[i] = zpos_new 

        # Return the updated position
        return zpos_new_arr
