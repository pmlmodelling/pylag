include "constants.pxi"

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from pylag.particle cimport Particle
from pylag.delta cimport Delta

# Base class for PositionModifier objects
cdef class PositionModifier:
    cdef void update_position(self, Particle *particle, Delta *delta_X) except *
