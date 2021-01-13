include "constants.pxi"

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from pylag.particle cimport Particle
from pylag.data_reader cimport DataReader


# Base class for PositionModifier objects
cdef class MortalityCalculator:

    cdef void apply(self, DataReader data_reader, DTYPE_FLOAT_t time,
                    Particle *particle) except *
