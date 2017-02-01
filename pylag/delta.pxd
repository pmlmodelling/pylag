# Data types used for constructing C data structures
from pylag.data_types_cython cimport DTYPE_FLOAT_t

cdef struct Delta:
    DTYPE_FLOAT_t x
    DTYPE_FLOAT_t y
    DTYPE_FLOAT_t z

cdef reset(Delta *delta)