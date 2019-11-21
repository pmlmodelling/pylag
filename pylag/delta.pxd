# Data types used for constructing C data structures
from pylag.data_types_cython cimport DTYPE_FLOAT_t

cdef struct Delta:
    DTYPE_FLOAT_t x1
    DTYPE_FLOAT_t x2
    DTYPE_FLOAT_t x3

cdef reset(Delta *delta)