from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef extern from "particle.cpp":
    pass

cdef extern from "particle.h" namespace "particles":
    cdef cppclass Particle:
        Particle() except +
        Particle(Particle&) except +
        Particle& operator=(Particle&) except +
        DTYPE_INT_t group_id
        DTYPE_INT_t id
        DTYPE_INT_t status
        DTYPE_FLOAT_t x1
        DTYPE_FLOAT_t x2
        DTYPE_FLOAT_t x3
        DTYPE_FLOAT_t phi[3]
        DTYPE_FLOAT_t omega_interfaces
        DTYPE_FLOAT_t omega_layers
        bint in_domain
        DTYPE_INT_t is_beached
        DTYPE_INT_t host_horizontal_elem
        DTYPE_INT_t k_layer
        bint in_vertical_boundary_layer
        DTYPE_INT_t k_lower_layer
        DTYPE_INT_t k_upper_layer
