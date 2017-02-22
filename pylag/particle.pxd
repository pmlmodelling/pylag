# Data types used for constructing C data structures
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# C Struct describing the physical location of a particle object
cdef struct Particle:
    # Particle properties
    # --------------------

    # Particle group ID
    DTYPE_INT_t group_id
    
    # Global coordinates
    # ------------------

    # Particle x-position
    DTYPE_FLOAT_t xpos
    
    # Particle y-position
    DTYPE_FLOAT_t ypos
    
    # Particle z-position
    DTYPE_FLOAT_t zpos

    # Local coordinates
    # -----------------

    # Barycentric coordinates within the host element
    DTYPE_FLOAT_t phi[3]

    # Vertical interpolation coefficient for variables defined at the interfaces
    # between k-levels
    DTYPE_FLOAT_t omega_interfaces

    # Indices describing the particle's position within a given grid
    # --------------------------------------------------------------
    
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
