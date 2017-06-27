include "constants.pxi"

from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle cimport Particle
from pylag.data_reader cimport DataReader
from pylag.delta cimport Delta
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator

# Base class for NumMethod objects
cdef class NumMethod:
    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR


# Base class for DetItMethod objects
cdef class DetItMethod:
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR


# Base class for StocItMethod objects
cdef class StocItMethod:
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR


# Base class for DetStocItMethod objects
cdef class DetStocItMethod:
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR