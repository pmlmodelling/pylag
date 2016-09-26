# Data types used for constructing C data structures
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef class Particle:
    cdef DTYPE_INT_t _group_id
    cdef DTYPE_FLOAT_t _xpos
    cdef DTYPE_FLOAT_t _ypos
    cdef DTYPE_FLOAT_t _zpos
    
    cdef DTYPE_INT_t _host_horizontal_elem

    cdef bint _in_domain
