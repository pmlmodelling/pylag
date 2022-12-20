include "constants.pxi"

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT

# PyLag cimports
from pylag.particle cimport Particle
from pylag.data_reader cimport DataReader


# Base class for windage velocity calculators
cdef class WindageCalculator:

    cdef void get_velocity(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t windage_velocity[2]) except *
