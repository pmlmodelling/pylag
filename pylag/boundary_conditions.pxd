from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from data_reader cimport DataReader
from particle cimport Particle

cdef class HorizBoundaryConditionCalculator:

    cdef apply(self, DataReader data_reader, Particle *particle_old,
            Particle *particle_new)

cdef class VertBoundaryConditionCalculator:

     cpdef DTYPE_FLOAT_t apply(self, DTYPE_FLOAT_t zpos, DTYPE_FLOAT_t zmin,
           DTYPE_FLOAT_t zmax)