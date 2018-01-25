include "constants.pxi"

from libcpp.vector cimport vector

from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef extern from "spline.h" namespace "tk":
    cdef cppclass spline:
        spline() except +
        void set_points(vector[DTYPE_FLOAT_t] x, vector[DTYPE_FLOAT_t] y)
        DTYPE_FLOAT_t operator()(DTYPE_FLOAT_t)
        DTYPE_FLOAT_t deriv(DTYPE_INT_t, DTYPE_FLOAT_t)

cdef class SplineWrapper:
    cdef spline c_spline

    cdef set_points(self, vector[DTYPE_FLOAT_t] x, vector[DTYPE_FLOAT_t] y)

    cdef DTYPE_FLOAT_t call(self, DTYPE_FLOAT_t x) except FLOAT_ERR
    
    cdef DTYPE_FLOAT_t deriv(self, DTYPE_INT_t order, DTYPE_FLOAT_t x) except FLOAT_ERR