# cython: profile=True
# cython: linetrace=True

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

cdef class NumIntegrator:
    cpdef advect(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader):
        pass
    
cdef class RK4Integrator(NumIntegrator):

    def __init__(self, time_step):
        self._time_step = time_step
    
        # Arrays for RK4 stages
        self.k1 = np.empty(3, dtype=DTYPE_FLOAT)
        self.k2 = np.empty(3, dtype=DTYPE_FLOAT)
        self.k3 = np.empty(3, dtype=DTYPE_FLOAT)
        self.k4 = np.empty(3, dtype=DTYPE_FLOAT)
        
        # Calculated vel
        self.vel = np.empty(3, dtype=DTYPE_FLOAT)
    
    cpdef advect(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader):
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
        data_reader.get_velocity(t, xpos, ypos, zpos, host, self.vel) 
        for i in xrange(ndim):
            self.k1[i] = self._time_step * self.vel[i]
        
        # Stage 2
        t = time + 0.5 * self._time_step
        xpos = particle.xpos + 0.5 * self.k1[0]
        ypos = particle.ypos + 0.5 * self.k1[1]
        zpos = particle.zpos + 0.5 * self.k1[2]
        host = data_reader.find_host(xpos, ypos, host)
        if host == -1:
            particle.host_horizontal_elem = -1
            particle.in_domain = -1
            return
        data_reader.get_velocity(t, xpos, ypos, zpos, host, self.vel) 
        for i in xrange(ndim):
            self.k2[i] = self._time_step * self.vel[i]

        # Stage 3
        t = time + 0.5 * self._time_step
        xpos = particle.xpos + 0.5 * self.k2[0]
        ypos = particle.ypos + 0.5 * self.k2[1]
        zpos = particle.zpos + 0.5 * self.k2[2]
        host = data_reader.find_host(xpos, ypos, host)
        if host == -1:
            particle.host_horizontal_elem = -1
            particle.in_domain = -1
            return
        data_reader.get_velocity(t, xpos, ypos, zpos, host, self.vel) 
        for i in xrange(ndim):
            self.k3[i] = self._time_step * self.vel[i]

        # Stage 4
        t = time + self._time_step
        xpos = particle.xpos + self.k3[0]
        ypos = particle.ypos + self.k3[1]
        zpos = particle.zpos + self.k3[2]
        host = data_reader.find_host(xpos, ypos, host)
        if host == -1:
            particle.host_horizontal_elem = -1
            particle.in_domain = -1
            return
        data_reader.get_velocity(t, xpos, ypos, zpos, host, self.vel) 
        for i in xrange(ndim):
            self.k4[i] = self._time_step * self.vel[i]

        # Update the particle's position
        particle.xpos = particle.xpos + (self.k1[0] + 2.0*self.k2[0] + 2.0*self.k3[0] + self.k4[0])/6.0
        particle.ypos = particle.ypos + (self.k1[1] + 2.0*self.k2[1] + 2.0*self.k3[1] + self.k4[1])/6.0
        particle.zpos = particle.zpos + (self.k1[2] + 2.0*self.k2[2] + 2.0*self.k3[2] + self.k4[2])/6.0
        particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos, particle.host_horizontal_elem)
        if particle.host_horizontal_elem == -1:
            particle.in_domain = -1
            return

def get_num_integrator(config):
    if config.get("PARTICLES", "num_integrator") == "RK4":
        return RK4Integrator(config.getfloat('PARTICLES', 'time_step'))
    else:
        raise ValueError('Unsupported numerical integration scheme.')   

