""" PyLag cwrappers

For speed, several of PyLag's Cython functions take c arrays or pointers as 
arguments. Cython does not appear to produce python wrappers for functions 
that have arguments with these types. The wrappers provided here are intended
to allow clients to call these functions in isolation.
"""

include "constants.pxi"

import numpy as np

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t
from pylag.integrator import get_num_integrator
from pylag.random_walk import get_vertical_random_walk_model

# PyLag cimports
cimport pylag.interpolation as interp
from pylag.data_reader cimport DataReader
from pylag.particle cimport Particle
from pylag.delta cimport Delta, reset
from pylag.integrator cimport NumIntegrator
from pylag.random_walk cimport VerticalRandomWalk

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

cpdef get_velocity(DataReader data_reader, t, x, y, z, host, zlayer):
    cdef DTYPE_FLOAT_t vel_c[N_VERTICES]
    cdef DTYPE_INT_t i
    
    data_reader.get_velocity(t, x, y, z, host, zlayer, vel_c)
    
    # Generate and pass back an array type python can understand
    vel_out = np.empty(N_VERTICES, dtype='float32')
    for i in xrange(N_VERTICES):
        vel_out[i] = vel_c[i]
    return vel_out

cpdef get_horizontal_velocity(DataReader data_reader, t, x, y, z, host, zlayer):
    cdef DTYPE_FLOAT_t vel_c[2]
    cdef DTYPE_INT_t i
    
    data_reader.get_horizontal_velocity(t, x, y, z, host, zlayer, vel_c)
    
    # Generate and pass back an array python can understand
    vel_out = np.empty(N_VERTICES, dtype='float32')
    for i in xrange(2):
        vel_out[i] = vel_c[i]
    return vel_out

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
        particle.host_z_layer = 0
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
    
    def __init__(self, config):
        
        self._vertical_random_walk = get_vertical_random_walk_model(config)
    
    def random_walk(self, data_reader, time, xpos, ypos, zpos):
        cdef Particle particle
        cdef Delta delta_X

        # Set these properties to default values
        particle.group_id = 0
        particle.host_horizontal_elem = 0
        particle.host_z_layer = 0
        particle.in_domain = True

        # Initialise remaining particle properties using the supplied arguments
        particle.xpos = xpos
        particle.ypos = ypos
        particle.zpos = zpos
        
        # Reset Delta object
        reset(&delta_X)
        
        # Advect the particle
        self._vertical_random_walk.random_walk(time, &particle, data_reader, &delta_X)

        # Apply reflecting surface/bottom boundary conditions
        zmin = data_reader.get_zmin(time, 0.0, 0.0)
        zmax = data_reader.get_zmax(time, 0.0, 0.0)
        if zpos < zmin:
            zpos = zmin + zmin - zpos
        elif zpos > zmax:
            zpos = zmax + zmax - zpos

        # Check for valid zpos
        if zpos < zmin:
            raise ValueError("New zpos (= {}) lies below the sea floor.".format(zpos))
        elif zpos > zmax:
            raise ValueError("New zpos (= {}) lies above the free surface.".format(zpos))          

        # Use Delta values to update the particle's position
        zpos_new = particle.zpos + delta_X.z

        # Return the updated position
        return zpos_new
