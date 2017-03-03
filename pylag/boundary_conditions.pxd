from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from data_reader cimport DataReader

cdef class HorizBoundaryConditionCalculator:

    cpdef apply(self, DataReader data_reader, DTYPE_FLOAT_t x_old,
            DTYPE_FLOAT_t y_old, DTYPE_FLOAT_t x_new, DTYPE_FLOAT_t y_new,
            DTYPE_INT_t last_host)

cdef class VertBoundaryConditionCalculator:

     cpdef DTYPE_FLOAT_t apply(self, DTYPE_FLOAT_t zpos, DTYPE_FLOAT_t zmin,
           DTYPE_FLOAT_t zmax)