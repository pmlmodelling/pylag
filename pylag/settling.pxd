include "constants.pxi"

from libcpp.string cimport string

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from pylag.data_reader cimport DataReader
from pylag.particle cimport Particle


cdef class SettlingVelocityCalculator:
    cdef void init_particle_settling_velocity(self, DataReader data_reader, DTYPE_FLOAT_t time,
                                              Particle *particle) except *

    cdef void set_particle_settling_velocity(self, DataReader data_reader, DTYPE_FLOAT_t time,
                                             Particle *particle) except *
