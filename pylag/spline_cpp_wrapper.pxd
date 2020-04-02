include "constants.pxi"

# Data types
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from libcpp.vector cimport vector

from spline cimport spline

cdef class SplineWrapper:
    cdef spline c_spline

    cdef set_points(self, vector[DTYPE_FLOAT_t] x, vector[DTYPE_FLOAT_t] y)

    cdef DTYPE_FLOAT_t call(self, DTYPE_FLOAT_t x) except FLOAT_ERR
    
    cdef DTYPE_FLOAT_t deriv(self, DTYPE_INT_t order, DTYPE_FLOAT_t x) except FLOAT_ERR
