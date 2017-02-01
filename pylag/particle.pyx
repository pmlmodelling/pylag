from cpython.mem cimport PyMem_Malloc, PyMem_Free

# Data types
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef class ParticleSmartPtr:
    """ Python object for managing the memory associated with Particle objects
    
    This class ties the lifetime of a Particle object allocated on the heap to 
    the lifetime of a ParticleSmartPtr object.
    """
    
    def __cinit__(self, DTYPE_INT_t group_id=-999, DTYPE_FLOAT_t xpos=-999., 
            DTYPE_FLOAT_t ypos=-999., DTYPE_FLOAT_t zpos=-999., DTYPE_INT_t host=-999, 
            DTYPE_INT_t host_z_layer=-999, DTYPE_INT_t in_domain=False):

        self._particle = <Particle *>PyMem_Malloc(sizeof(Particle))

        if not self._particle:
            raise MemoryError()

        self._particle.group_id = group_id
        self._particle.xpos = xpos
        self._particle.ypos = ypos
        self._particle.zpos = zpos
        self._particle.host_horizontal_elem = host
        self._particle.host_z_layer = host_z_layer
        self._particle.in_domain = in_domain

    def __dealloc__(self):
        PyMem_Free(self._particle)

    cdef Particle* get_ptr(self):
        return self._particle

cdef ParticleSmartPtr copy(ParticleSmartPtr particle_smart_ptr):
    """ Create a copy of a ParticleSmartPtr object
    
    This function creates a new copy a ParticleSmartPtr object. In so doing
    new memory is allocated. This memory is automatically freed when the
    ParticleSmartPtr is deleted.
    
    Parameters:
    -----------
    particle_smart_ptr : ParticleSmartPtr
        ParticleSmartPtr object.
    
    Returns:
    --------
    particle_smart_ptr : ParticleSmartPtr
        An exact copy of the ParticleSmartPtr object passed in.
    """
    cdef Particle* particle_ptr
    
    particle_ptr = particle_smart_ptr.get_ptr()

    return ParticleSmartPtr(particle_ptr.group_id,
                            particle_ptr.xpos,
                            particle_ptr.ypos,
                            particle_ptr.zpos,
                            particle_ptr.host_horizontal_elem,
                            particle_ptr.host_z_layer,
                            particle_ptr.in_domain)
