# Data types used for constructing C data structures
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# C Struct describing the physical location of a particle object
cdef struct Particle:
    # Particle group ID
    DTYPE_INT_t group_id
    
    # Particle x-position
    DTYPE_FLOAT_t xpos
    
    # Particle y-position
    DTYPE_FLOAT_t ypos
    
    # Particle z-position
    DTYPE_FLOAT_t zpos
    
    # The host horizontal element
    DTYPE_INT_t host_horizontal_elem

    # The host z layer
    DTYPE_INT_t host_z_layer

    # Flag identifying whether or not the particle resides within the model domain.
    bint in_domain

cdef class ParticleSmartPtr:
    cdef Particle* _particle

    cdef Particle* get_ptr(self)

cdef ParticleSmartPtr copy(ParticleSmartPtr)
