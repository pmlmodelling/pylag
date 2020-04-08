from cython.operator cimport dereference as deref

from libcpp.vector cimport vector

# Data types
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t


cdef class ParticleSmartPtr:
    """ Python object for managing the memory associated with Particle objects
    
    This class ties the lifetime of a Particle object allocated on the heap to 
    the lifetime of a ParticleSmartPtr object.
    """
    
    def __cinit__(self, DTYPE_INT_t group_id=-999, DTYPE_FLOAT_t x1=-999., 
            DTYPE_FLOAT_t x2=-999., DTYPE_FLOAT_t x3=-999.,
            DTYPE_FLOAT_t phi1=-999.,  DTYPE_FLOAT_t phi2=-999.,
            DTYPE_FLOAT_t phi3=-999., DTYPE_FLOAT_t omega_interfaces=-999.,
            DTYPE_FLOAT_t omega_layers=-999., bint in_domain=False,
            DTYPE_INT_t is_beached=0, DTYPE_INT_t host=-999, 
            DTYPE_INT_t k_layer=-999, bint in_vertical_boundary_layer=False,
            DTYPE_INT_t k_lower_layer=-999, DTYPE_INT_t k_upper_layer=-999,
            DTYPE_INT_t id=-999, DTYPE_INT_t status=0, ParticleSmartPtr particle_smart_ptr=None):

        cdef ParticleSmartPtr _particle_smart_ptr
        cdef vector[DTYPE_FLOAT_t] _phi

        # Call copy ctor if particle_smart_ptr is given. Else, use default ctor
        if particle_smart_ptr and type(particle_smart_ptr) is ParticleSmartPtr:
            _particle_smart_ptr = <ParticleSmartPtr> particle_smart_ptr
            self._particle = new Particle(deref(_particle_smart_ptr._particle))

        else:
            self._particle = new Particle()

            # Overwrite with supplied optional arguments
            self._particle.group_id = group_id
            self._particle.x1 = x1
            self._particle.x2 = x2
            self._particle.x3 = x3
            self._particle.omega_interfaces = omega_interfaces
            self._particle.omega_layers = omega_layers
            self._particle.in_domain = in_domain
            self._particle.is_beached = is_beached
            self._particle.id = id
            self._particle.status = status
            self._particle.set_phi([phi1, phi2, phi3])
            self._particle.set_host_horizontal_elem(host)
            self._particle.set_k_layer(k_layer)
            self._particle.set_in_vertical_boundary_layer(in_vertical_boundary_layer)
            self._particle.set_k_lower_layer(k_lower_layer)
            self._particle.set_k_upper_layer(k_upper_layer)

        if not self._particle:
            raise MemoryError()

    def __dealloc__(self):
        del self._particle

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
                'x1': self._particle.x1,
                'x2': self._particle.x2,
                'x3': self._particle.x3}

        return data

    @property
    def x1(self):
        return self._particle.x1

    @property
    def x2(self):
        return self._particle.x2

    @property
    def x3(self):
        return self._particle.x3

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
    def host_horizontal_elem(self):
        return self._particle.get_host_horizontal_elem()

    @property
    def k_layer(self):
        return self._particle.get_k_layer()

    @property
    def k_lower_layer(self):
        return self._particle.get_k_lower_layer()

    @property
    def k_upper_layer(self):
        return self._particle.get_k_upper_layer()

    @property
    def in_vertical_boundary_layer(self):
        return self._particle.get_in_vertical_boundary_layer()

    @property
    def phi(self):
        phi = []
        for x in self._particle.get_phi():
            phi.append(x)
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

    return ParticleSmartPtr(particle_smart_ptr=particle_smart_ptr)


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
    cdef vector[DTYPE_FLOAT_t] phi = particle.get_phi()

    s = "Particle properties \n"\
        "------------------- \n"\
        "Particle id = {} \n"\
        "Particle x1 = {} \n"\
        "Particle x2 = {} \n"\
        "Particle x3 = {} \n"\
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
                                             particle.x1,
                                             particle.x2,
                                             particle.x3,
                                             phi[0],
                                             phi[1],
                                             phi[2],
                                             particle.omega_interfaces,
                                             particle.omega_layers,
                                             particle.get_host_horizontal_elem(),
                                             particle.get_in_vertical_boundary_layer(),
                                             particle.get_k_layer(),
                                             particle.get_k_lower_layer(),
                                             particle.get_k_upper_layer(),
                                             particle.in_domain,
                                             particle.is_beached)

    return s
