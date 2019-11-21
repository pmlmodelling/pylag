include "constants.pxi"

from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from pylag.particle cimport Particle
from pylag.data_reader cimport DataReader
from pylag.delta cimport Delta

# Base class for NumMethod objects
cdef class NumMethod:
    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle) except INT_ERR


# Base class for ItMethod objects
cdef class ItMethod:
    cdef DTYPE_FLOAT_t _time_step

    cdef DTYPE_FLOAT_t _time_direction

    cdef DTYPE_FLOAT_t get_time_step(self)

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR
