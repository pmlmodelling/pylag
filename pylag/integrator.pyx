# cython: profile=True
# cython: linetrace=True

import numpy as np

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

class NumIntegrator(object):
    def advect(self, time, particle, data_reader):
        pass
    
class RK4Integrator(NumIntegrator):
    def __init__(self, time_step):
        self._time_step = time_step
    
    def advect(self, time, particle, data_reader):
        # Arrays for RK4 stages
        cdef DTYPE_FLOAT_t[:] k1 = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] k2 = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] k3 = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] k4 = np.empty(3, dtype=DTYPE_FLOAT)
        
        # Temporary containers
        cdef DTYPE_FLOAT_t t, xpos, ypos, zpos
        cdef DTYPE_INT_t host
        cdef DTYPE_FLOAT_t[:] vel
        
        # Array indices/loop counters
        cdef DTYPE_INT_t ndim = 3
        cdef DTYPE_INT_t i

        vel = np.empty(ndim, dtype=DTYPE_FLOAT)
        
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
        host = data_reader.find_host(xpos, ypos, host)
        if host == -1:
            particle.host_horizontal_elem = -1
            particle.in_domain = -1
            return
        data_reader.get_velocity(t, xpos, ypos, zpos, host, vel) 
        for i in xrange(ndim):
            k2[i] = self._time_step * vel[i]

        # Stage 3
        t = time + 0.5 * self._time_step
        xpos = particle.xpos + 0.5 * k2[0]
        ypos = particle.ypos + 0.5 * k2[1]
        zpos = particle.zpos + 0.5 * k2[2]
        host = data_reader.find_host(xpos, ypos, host)
        if host == -1:
            particle.host_horizontal_elem = -1
            particle.in_domain = -1
            return
        data_reader.get_velocity(t, xpos, ypos, zpos, host, vel) 
        for i in xrange(ndim):
            k3[i] = self._time_step * vel[i]

        # Stage 4
        t = time + self._time_step
        xpos = particle.xpos + k3[0]
        ypos = particle.ypos + k3[1]
        zpos = particle.zpos + k3[2]
        host = data_reader.find_host(xpos, ypos, host)
        if host == -1:
            particle.host_horizontal_elem = -1
            particle.in_domain = -1
            return
        data_reader.get_velocity(t, xpos, ypos, zpos, host, vel) 
        for i in xrange(ndim):
            k4[i] = self._time_step * vel[i]

        # Update the particle's position
        particle.xpos = particle.xpos + (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])/6.0
        particle.ypos = particle.ypos + (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])/6.0
        particle.zpos = particle.zpos + (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])/6.0
        particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos, particle.host_horizontal_elem)
        if particle.host_horizontal_elem == -1:
            particle.in_domain = -1
            return

def get_num_integrator(config):
    if config.get("PARTICLES", "num_integrator") == "RK4":
        return RK4Integrator(config.getfloat('PARTICLES', 'time_step'))
    else:
        raise ValueError('Unsupported numerical integration scheme.')   

