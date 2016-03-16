# Data types used for constructing C data structures
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef class Delta:
    cdef DTYPE_FLOAT_t _delta_x
    cdef DTYPE_FLOAT_t _delta_y
    cdef DTYPE_FLOAT_t _delta_z

    cpdef reset(self)