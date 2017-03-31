from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle cimport Particle
from pylag.data_reader cimport DataReader
from pylag.delta cimport Delta
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator

# Base class for Lagrangian Stochastic Models (LSMs)
cdef class LSM:
    cdef DTYPE_FLOAT_t _time_step

    cdef apply(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)


# LSM for computing vertical movement
cdef class OneDLSM(LSM):
    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    cdef apply(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)

# LSM for computing horizontal movement
cdef class HorizontalLSM(LSM):

    cdef apply(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)
