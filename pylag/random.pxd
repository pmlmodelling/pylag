include "constants.pxi"

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cpdef DTYPE_FLOAT_t gauss(DTYPE_FLOAT_t mean = ?, DTYPE_FLOAT_t std = ?) except FLOAT_ERR

cpdef DTYPE_FLOAT_t uniform(DTYPE_FLOAT_t a = ?, DTYPE_FLOAT_t b = ?) except FLOAT_ERR