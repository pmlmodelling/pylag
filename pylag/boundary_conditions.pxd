include "constants.pxi"

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from data_reader cimport DataReader
from particle cimport Particle

cdef class HorizBoundaryConditionCalculator:

    cdef DTYPE_INT_t apply(self, DataReader data_reader, Particle *particle_old,
                           Particle *particle_new) except INT_ERR

cdef class VertBoundaryConditionCalculator:

     cdef DTYPE_INT_t apply(self, DataReader data_reader, DTYPE_FLOAT_t time, 
                            Particle *particle) except INT_ERR
