from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle import Particle
from particle cimport Particle
from data_reader import DataReader
from data_reader cimport DataReader
from delta import Delta
from delta cimport Delta

cdef class NumIntegrator:
    cpdef advect(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X)

cdef class RK4Integrator2D(NumIntegrator):
    cdef DTYPE_FLOAT_t _time_step

    cpdef advect(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X)
    
cdef class RK4Integrator3D(NumIntegrator):
    cdef DTYPE_FLOAT_t _time_step

    # Grid boundary limits
    cdef DTYPE_FLOAT_t _zmin
    cdef DTYPE_FLOAT_t _zmax

    cpdef advect(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X)
