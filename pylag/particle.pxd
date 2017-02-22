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

    # Vertical interpolation coefficient for variables defined at the mid-point
    # of l-layers
    DTYPE_FLOAT_t omega_layers

    # Indices describing the particle's position within a given grid
    # --------------------------------------------------------------

    # Flag identifying whether or not the particle resides within the model domain.
    bint in_domain

    # The host horizontal element
    DTYPE_INT_t host_horizontal_elem

    # The host z layer
    DTYPE_INT_t host_z_layer

    # Flag for whether the particle is in the top or bottom boundary layers
    bint in_vertical_boundary_layer

    # Index of the k-layer lying immediately below the particle's current
    # position. Only set if the particle is not in the top or bottom boundary
    # layers
    DTYPE_INT_t k_lower_layer

    # Index of the k-layer lying immediately above the particle's current
    # position. Only set if the particle is not in the top or bottom boundary
    # layers
    DTYPE_INT_t k_upper_layer

cdef class ParticleSmartPtr:
    cdef Particle* _particle

    cdef Particle* get_ptr(self)

cdef ParticleSmartPtr copy(ParticleSmartPtr)
