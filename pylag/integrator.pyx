import logging
import numpy as np

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle cimport Particle
from data_reader cimport DataReader
from delta cimport Delta

cdef class NumIntegrator:
    cpdef advect(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X):
        pass

cdef class RK4Integrator2D(NumIntegrator):

    def __init__(self, config):
        self._time_step = config.getfloat('SIMULATION', 'time_step')
    
    cpdef advect(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X):
        """
        Advect particles forward in time. If particles are advected outside of
        the model domain, the particle's position is not updated. This mimics
        the behaviour of FVCOM's particle tracking model at solid boundaries.
        """
        # Arrays for RK4 stages
        cdef DTYPE_FLOAT_t k1[2]
        cdef DTYPE_FLOAT_t k2[2]
        cdef DTYPE_FLOAT_t k3[2]
        cdef DTYPE_FLOAT_t k4[2]

        # Calculated vel
        cdef DTYPE_FLOAT_t vel[2]
        
        # Temporary containers
        cdef DTYPE_FLOAT_t t, xpos, ypos, zpos
        cdef DTYPE_INT_t host

        # Array indices/loop counters
        cdef DTYPE_INT_t ndim = 2
        cdef DTYPE_INT_t i
        
        # Stage 1
        t = time
        xpos = particle.xpos
        ypos = particle.ypos
        zpos = particle.zpos
        host = particle.host_horizontal_elem
        data_reader.get_horizontal_velocity(t, xpos, ypos, zpos, host, vel) 
        for i in xrange(ndim):
            k1[i] = self._time_step * vel[i]
        
        # Stage 2
        t = time + 0.5 * self._time_step
        xpos = particle.xpos + 0.5 * k1[0]
        ypos = particle.ypos + 0.5 * k1[1]
        
        host = data_reader.find_host(xpos, ypos, host)
        if host == -1: return
        data_reader.get_horizontal_velocity(t, xpos, ypos, zpos, host, vel) 
        for i in xrange(ndim):
            k2[i] = self._time_step * vel[i]

        # Stage 3
        t = time + 0.5 * self._time_step
        xpos = particle.xpos + 0.5 * k2[0]
        ypos = particle.ypos + 0.5 * k2[1]
        
        host = data_reader.find_host(xpos, ypos, host)
        if host == -1: return
        data_reader.get_horizontal_velocity(t, xpos, ypos, zpos, host, vel) 
        for i in xrange(ndim):
            k3[i] = self._time_step * vel[i]

        # Stage 4
        t = time + self._time_step
        xpos = particle.xpos + k3[0]
        ypos = particle.ypos + k3[1]

        host = data_reader.find_host(xpos, ypos, host)
        if host == -1: return
        data_reader.get_horizontal_velocity(t, xpos, ypos, zpos, host, vel) 
        for i in xrange(ndim):
            k4[i] = self._time_step * vel[i]

        # Sum changes and save
        delta_X.x += (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])/6.0
        delta_X.y += (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])/6.0
    
cdef class RK4Integrator3D(NumIntegrator):

    def __init__(self, config):
        self._time_step = config.getfloat('SIMULATION', 'time_step')
        self._zmin = config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmin')
        self._zmax = config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmax')
    
    cpdef advect(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X):
        """
        Advect particles forward in time. If particles are advected outside of
        the model domain, the particle's position is not updated. This mimics
        the behaviour of FVCOM's particle tracking model at solid boundaries.
        
        TODO - this is not an ideal solution. Firstly, open boundaries are not
        distinguished from solid boundaries, and it makes more sense for
        particles to be lost from the model domain when they cross an open
        boundary. And secondly, it seems like we should be able to do something
        better at solid boundaries.
        """
        # Arrays for RK4 stages
        cdef DTYPE_FLOAT_t k1[3]
        cdef DTYPE_FLOAT_t k2[3]
        cdef DTYPE_FLOAT_t k3[3]
        cdef DTYPE_FLOAT_t k4[3]

        # Calculated vel
        cdef DTYPE_FLOAT_t vel[3]
        
        # Temporary containers
        cdef DTYPE_FLOAT_t t, xpos, ypos, zpos
        cdef DTYPE_INT_t host

        # Array indices/loop counters
        cdef DTYPE_INT_t ndim = 3
        cdef DTYPE_INT_t i
        
        # Stage 1
        t = time
        xpos = particle.xpos
        ypos = particle.ypos
        zpos = particle.zpos
        host = particle.host_horizontal_elem
        data_reader.get_velocity(t, xpos, ypos, zpos, host, vel) 
        for i in xrange(ndim):
            k1[i] = self._time_step * vel[i]
        
        # Stage 2
        t = time + 0.5 * self._time_step
        xpos = particle.xpos + 0.5 * k1[0]
        ypos = particle.ypos + 0.5 * k1[1]
        zpos = particle.zpos + 0.5 * k1[2]
        
        # Impose reflecting boundary condition in z
        if zpos < self._zmin:
            zpos = self._zmin + self._zmin - zpos
        elif zpos > self._zmax:
            zpos = self._zmax + self._zmax - zpos
        
        host = data_reader.find_host(xpos, ypos, host)
        if host == -1: return
        data_reader.get_velocity(t, xpos, ypos, zpos, host, vel) 
        for i in xrange(ndim):
            k2[i] = self._time_step * vel[i]

        # Stage 3
        t = time + 0.5 * self._time_step
        xpos = particle.xpos + 0.5 * k2[0]
        ypos = particle.ypos + 0.5 * k2[1]
        zpos = particle.zpos + 0.5 * k2[2]
        
        # Impose reflecting boundary condition in z
        if zpos < self._zmin:
            zpos = self._zmin + self._zmin - zpos
        elif zpos > self._zmax:
            zpos = self._zmax + self._zmax - zpos
        
        host = data_reader.find_host(xpos, ypos, host)
        if host == -1: return
        data_reader.get_velocity(t, xpos, ypos, zpos, host, vel) 
        for i in xrange(ndim):
            k3[i] = self._time_step * vel[i]

        # Stage 4
        t = time + self._time_step
        xpos = particle.xpos + k3[0]
        ypos = particle.ypos + k3[1]
        zpos = particle.zpos + k3[2]
        
        # Impose reflecting boundary condition in z
        if zpos < self._zmin:
            zpos = self._zmin + self._zmin - zpos
        elif zpos > self._zmax:
            zpos = self._zmax + self._zmax - zpos

        host = data_reader.find_host(xpos, ypos, host)
        if host == -1: return
        data_reader.get_velocity(t, xpos, ypos, zpos, host, vel) 
        for i in xrange(ndim):
            k4[i] = self._time_step * vel[i]

        # Sum changes and save
        delta_X.x += (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])/6.0
        delta_X.y += (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])/6.0
        delta_X.z += (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])/6.0

def get_num_integrator(config):
    if not config.has_option("SIMULATION", "num_integrator"):
        logger = logging.getLogger(__name__)
        logger.info('Configuation option num_integrator not found. The model '\
            'will run without advection.')
        return None
    
    # Return the specified numerical integrator.
    if config.get("SIMULATION", "num_integrator") == "RK4_2D":
        return RK4Integrator2D(config)
    if config.get("SIMULATION", "num_integrator") == "RK4_3D":
        return RK4Integrator3D(config)
    elif config.get("SIMULATION", "num_integrator") == "None":
        return None
    else:
        raise ValueError('Unsupported numerical integration scheme.')

