from cpython.mem cimport PyMem_Malloc, PyMem_Free

# Data types
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef class ParticleSmartPtr:
    """ Python object for managing the memory associated with Particle objects
    
    This class ties the lifetime of a Particle object allocated on the heap to 
    the lifetime of a ParticleSmartPtr object.
    """
    
    def __cinit__(self, DTYPE_INT_t group_id=-999, DTYPE_FLOAT_t xpos=-999., 
            DTYPE_FLOAT_t ypos=-999., DTYPE_FLOAT_t zpos=-999.,
            DTYPE_FLOAT_t phi1=-999.,  DTYPE_FLOAT_t phi2=-999.,
            DTYPE_FLOAT_t phi3=-999., DTYPE_FLOAT_t omega_interfaces=-999.,
            DTYPE_FLOAT_t omega_layers=-999., DTYPE_INT_t in_domain=False,
            DTYPE_INT_t is_beached=0, DTYPE_INT_t host=-999, 
            DTYPE_INT_t k_layer=-999, DTYPE_INT_t in_vertical_boundary_layer=False,
            DTYPE_INT_t k_lower_layer=-999, DTYPE_INT_t k_upper_layer=-999,
            DTYPE_INT_t id=-999, DTYPE_INT_t status=0):

        self._particle = <Particle *>PyMem_Malloc(sizeof(Particle))

        if not self._particle:
            raise MemoryError()

        self._particle.group_id = group_id
        self._particle.xpos = xpos
        self._particle.ypos = ypos
        self._particle.zpos = zpos
        self._particle.phi[0] = phi1
        self._particle.phi[1] = phi2
        self._particle.phi[2] = phi3
        self._particle.omega_interfaces = omega_interfaces
        self._particle.omega_layers = omega_layers
        self._particle.in_domain = in_domain
        self._particle.is_beached = is_beached
        self._particle.host_horizontal_elem = host
        self._particle.k_layer = k_layer
        self._particle.in_vertical_boundary_layer = in_vertical_boundary_layer
        self._particle.k_lower_layer = k_lower_layer
        self._particle.k_upper_layer = k_upper_layer
        self._particle.id = id
        self._particle.status = status

    def __dealloc__(self):
        PyMem_Free(self._particle)

    cdef Particle* get_ptr(self):
        return self._particle

    def get_particle_data(self):
        """ Get particle data

        Return data describing the particle's basic state. Do not return
        any derived data (e.g. particle local coordinates). The purpose
        is to return just enough data to make it possible to recreate
        the particle at some later time (e.g. from a restart file).

        Returns:
        --------
        data : dict
            Dictionary containing data that describes the particle's basic state.
        """

        data = {'group_id': self._particle.group_id,
                'x1': self._particle.xpos,
                'x2': self._particle.ypos,
                'x3': self._particle.zpos}

        return data

    @property
    def xpos(self):
        return self._particle.xpos

    @property
    def ypos(self):
        return self._particle.ypos

    @property
    def zpos(self):
        return self._particle.zpos

    @property
    def host_horizontal_elem(self):
        return self._particle.host_horizontal_elem

    @property
    def omega_interfaces(self):
        return self._particle.omega_interfaces

    @property
    def omega_layers(self):
        return self._particle.omega_layers

    @property
    def in_domain(self):
        return self._particle.in_domain

    @property
    def is_beached(self):
        return self._particle.is_beached

    @property
    def k_layer(self):
        return self._particle.k_layer

    @property
    def k_lower_layer(self):
        return self._particle.k_lower_layer

    @property
    def k_upper_layer(self):
        return self._particle.k_upper_layer

    @property
    def in_vertical_boundary_layer(self):
        return self._particle.in_vertical_boundary_layer

    @property
    def phi(self):
        phi = []
        for i in xrange(3):
            phi.append(self._particle.phi[i])
        return phi

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
                            particle_ptr.phi[0],
                            particle_ptr.phi[1],
                            particle_ptr.phi[2],
                            particle_ptr.omega_interfaces,
                            particle_ptr.omega_layers,
                            particle_ptr.in_domain,
                            particle_ptr.is_beached,
                            particle_ptr.host_horizontal_elem,
                            particle_ptr.k_layer,
                            particle_ptr.in_vertical_boundary_layer,
                            particle_ptr.k_lower_layer,
                            particle_ptr.k_upper_layer,
                            particle_ptr.id,
                            particle_ptr.status)

cdef to_string(Particle* particle):
    """ Return a string object that describes a particle
    
    Parameters:
    -----------
    particle : Particle C ptr
        Pointer to a particle object

    Returns:
    --------
    s : str
        String describing the particle
    """
    s = "Particle properties \n"\
        "------------------- \n"\
        "Particle id = {} \n"\
        "Particle xpos = {} \n"\
        "Particle ypos = {} \n"\
        "Particle zpos = {} \n"\
        "Particle phi[0] = {} \n"\
        "Particle phi[1] = {} \n"\
        "Particle phi[2] = {} \n"\
        "Particle omega interfaces = {} \n"\
        "Particle omega layers = {} \n"\
        "Particle host element = {} \n"\
        "Partilce in vertical boundary layer = {} \n"\
        "Partilce k layer = {} \n"\
        "Partilce k lower layer = {} \n"\
        "Partilce k upper layer = {} \n"\
        "Particle in domain = {} \n"\
        "Particle is beached = {} \n".format(particle.id, 
                                             particle.xpos,
                                             particle.ypos,
                                             particle.zpos,
                                             particle.phi[0],
                                             particle.phi[1],
                                             particle.phi[2],
                                             particle.omega_interfaces,
                                             particle.omega_layers,
                                             particle.host_horizontal_elem,
                                             particle.in_vertical_boundary_layer,
                                             particle.k_layer,
                                             particle.k_lower_layer,
                                             particle.k_upper_layer,
                                             particle.in_domain,
                                             particle.is_beached)
                                             
    return s
