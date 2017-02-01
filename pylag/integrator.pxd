from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle cimport Particle
from data_reader cimport DataReader
from delta cimport Delta

cdef class NumIntegrator:
    cdef DTYPE_INT_t advect(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X)
