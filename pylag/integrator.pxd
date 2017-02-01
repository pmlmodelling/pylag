from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from pylag.particle cimport Particle
from pylag.data_reader cimport DataReader
from pylag.delta cimport Delta

cdef class NumIntegrator:
    cdef DTYPE_INT_t advect(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X)
