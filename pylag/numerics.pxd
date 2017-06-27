from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle cimport Particle
from pylag.data_reader cimport DataReader
from pylag.delta cimport Delta
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator

# Base class for NumMethod objects
cdef class NumMethod:

    cdef step(self, DataReader data_reader, DTYPE_FLOAT_t time, Particle *particle, Delta *delta_X)
