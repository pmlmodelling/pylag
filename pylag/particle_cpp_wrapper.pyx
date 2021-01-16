"""
This module provides a Python friendly API for interacting with
Particle objects. The code for the latter is implemented in C++.
Particle objects include a set of attributes that describe their
state and location within a given domain.
"""

from cython.operator cimport dereference as deref

from libcpp.vector cimport vector
from libcpp.string cimport string

# Data types
from pylag.data_types_python import INT_INVALID, FLOAT_INVALID

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef class ParticleSmartPtr:
    """ Python object for managing the memory associated with Particle objects
    
    This class ties the lifetime of a Particle object allocated on the heap to 
    the lifetime of a ParticleSmartPtr object.
    """
    
    def __cinit__(self, DTYPE_INT_t group_id=INT_INVALID, DTYPE_FLOAT_t x1=FLOAT_INVALID,
                  DTYPE_FLOAT_t x2=FLOAT_INVALID, DTYPE_FLOAT_t x3=FLOAT_INVALID, phis={},
                  DTYPE_FLOAT_t omega_interfaces=FLOAT_INVALID,
                  DTYPE_FLOAT_t omega_layers=FLOAT_INVALID, bint in_domain=False,
                  DTYPE_INT_t is_beached=0, host_elements={},
                  DTYPE_INT_t k_layer=INT_INVALID, bint in_vertical_boundary_layer=False,
                  DTYPE_INT_t k_lower_layer=INT_INVALID, DTYPE_INT_t k_upper_layer=INT_INVALID,
                  DTYPE_INT_t id=INT_INVALID, DTYPE_INT_t status=0, DTYPE_FLOAT_t age=FLOAT_INVALID,
                  bint is_alive=False, bio_parameters={}, ParticleSmartPtr particle_smart_ptr=None):

        cdef ParticleSmartPtr _particle_smart_ptr

        # Call copy ctor if particle_smart_ptr is given. Else, use default ctor
        if particle_smart_ptr and type(particle_smart_ptr) is ParticleSmartPtr:
            _particle_smart_ptr = <ParticleSmartPtr> particle_smart_ptr
            self._particle = new Particle(deref(_particle_smart_ptr._particle))

        else:
            # TODO drop use of default constructor here and make it private?
            self._particle = new Particle()

            # Overwrite with supplied optional arguments
            self._particle.set_group_id(group_id)
            self._particle.set_id(id)
            self._particle.set_status(status)
            self._particle.set_x1(x1)
            self._particle.set_x2(x2)
            self._particle.set_x3(x3)
            self._particle.set_omega_interfaces(omega_interfaces)
            self._particle.set_omega_layers(omega_layers)
            self._particle.set_in_domain(in_domain)
            self._particle.set_is_beached(is_beached)
            self._particle.set_k_layer(k_layer)
            self._particle.set_in_vertical_boundary_layer(in_vertical_boundary_layer)
            self._particle.set_k_lower_layer(k_lower_layer)
            self._particle.set_k_upper_layer(k_upper_layer)
            self._particle.set_age(age)
            self._particle.set_is_alive(is_alive)

            # Set local coordinates on all grids
            self.set_all_phis(phis)

            # Add hosts
            self.set_all_host_horizontal_elems(host_elements)

            # Set all bio parameters
            self.set_all_bio_parameters(bio_parameters)

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

        Returns
        -------
        data : dict
            Dictionary containing data that describes the particle's basic state.
        """

        data = {'group_id': self._particle.get_group_id(),
                'x1': self._particle.get_x1(),
                'x2': self._particle.get_x2(),
                'x3': self._particle.get_x3()}

        return data

    def set_phi(self, grid, phi):
        """ Set local coordinates for the given grid

        Parameters
        ----------
        grid : str
            The name of the grid.

        phi : 1D NumPy array
            1D array of particle local coordinates.
        """
        grid_name = grid.encode() if type(grid) == str else grid
        self._particle.set_phi(grid_name, phi)

    def get_phi(self, grid):
        """ Get local coordinates for the given grid

        Parameters
        ----------
        grid : str
            The name of the grid.

        Returns
        -------
        phi : 1D NumPy array
            Particle local coordinates for the given grid.
        """
        grid_name = grid.encode() if type(grid) == str else grid

        phi = []
        for x in self._particle.get_phi(grid_name):
            phi.append(x)
        return phi

    def set_all_phis(self, phis):
        """ Set local coordinates for all grids

        Parameters
        ----------
        phis : dict
            Dictionary of local coords {grid_name: [phi1, phi2, [phi3]}
        """
        self._particle.clear_phis()
        for grid, phi in phis.items():
            self.set_phi(grid, phi)

    def set_host_horizontal_elem(self, grid, host):
        """ Set host element for the given grid

        Parameters
        ----------
        grid : str
            The name of the grid.

        host : int
            The host element
        """
        grid_name = grid.encode() if type(grid) == str else grid
        self._particle.set_host_horizontal_elem(grid_name, host)

    def get_host_horizontal_elem(self, grid):
        """ Get host element for the given grid

        Parameters
        ----------
        grid : str
            The name of the grid.

        Returns
        -------
         : int
             The host element.
        """
        grid_name = grid.encode() if type(grid) == str else grid
        return self._particle.get_host_horizontal_elem(grid_name)

    def set_all_host_horizontal_elems(self, host_elements):
        """ Set host elements for all grids

        Parameters
        ----------
        host_elements : dict
            Dictionary of host elements {grid_name: host}
        """
        self._particle.clear_host_horizontal_elems()
        for grid, host in host_elements.items():
            self.set_host_horizontal_elem(grid, host)

    def get_all_host_horizontal_elems(self):
        """ Get all host elements for the given grid

        Returns
        -------
        host_elements : dict
             The host elements stored in a dictionary by grid name.
        """
        cdef vector[string] grids
        cdef vector[int] hosts
        self._particle.get_all_host_horizontal_elems(grids, hosts)

        host_elements = {}
        for grid, host in zip(grids, hosts):
            host_elements[grid.decode()] = host

        return host_elements

    def set_age(self, age):
        """ Set the particle's age

        Parameters
        ----------
        age : float
            The age in seconds.
        """
        self._particle.set_age(age)

    def set_bio_parameter(self, name, value):
        """ Set biological parameter

        Parameters
        ----------
        name : str
            The name of the parameter.

        value : float
            The parameters value.
        """
        param_name = name.encode() if type(name) == str else name
        self._particle.set_bio_parameter(param_name, value)

    def get_bio_parameter(self, name):
        """ Get biological parameter

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        value : float
            The value of the parameter.
        """
        param_name = name.encode() if type(name) == str else name

        return self._particle.get_bio_parameter(param_name)

    def set_all_bio_parameters(self, bio_parameters):
        """ Set all bio parameters

        Parameters
        ----------
        bio_parameters : dict
            Dictionary of bio parameters {name: value}
        """
        self._particle.clear_bio_parameters()
        for name, value in bio_parameters.items():
            self.set_phi(name, value)

    def get_all_bio_parameters(self):
        """ Get all bio parameters

        Returns
        -------
        bio_parameters : dict
             Bio parameters stored in a dictionary.
        """
        cdef vector[string] names
        cdef vector[float] values
        self._particle.get_all_bio_parameters(names, values)

        bio_parameters = {}
        for name, value in zip(names, values):
            bio_parameters[name.decode()] = value

        return bio_parameters

    @property
    def status(self):
        """ The particle's status (0 - okay; 1 - error) """
        return self._particle.get_status()

    @property
    def x1(self):
        """ The particle's x1-coordinate """
        return self._particle.get_x1()

    @property
    def x2(self):
        """ The particle's x2-coordinate """
        return self._particle.get_x2()

    @property
    def x3(self):
        """ The particle's x1-coordinate """
        return self._particle.get_x3()

    @property
    def omega_interfaces(self):
        """ Vertical interpolation coefficient for variables defined at the interfaces between k-levels """
        return self._particle.get_omega_interfaces()

    @property
    def omega_layers(self):
        """ Vertical interpolation coefficient for variables defined at the interfaces between k-layers """
        return self._particle.get_omega_layers()

    @property
    def in_domain(self):
        """ Flag identifying whether or not the particle resides within the model domain (1 - yes; 0 - no) """
        return self._particle.get_in_domain()

    @property
    def is_beached(self):
        """ Flag identifying whether or not the particle is beached (1 - yes; 0 - no) """
        return self._particle.get_is_beached()

    @property
    def k_layer(self):
        """ Index identifying the k-layer the particle currently resides in """
        return self._particle.get_k_layer()

    @property
    def k_lower_layer(self):
        """ Index identifying the k-level immediately below the particle """
        return self._particle.get_k_lower_layer()

    @property
    def k_upper_layer(self):
        """ Index identifying the k-level immediately above the particle """
        return self._particle.get_k_upper_layer()

    @property
    def in_vertical_boundary_layer(self):
        """ Flag signifying whether or not the particle resides in either the top or bottom boundary layers """
        return self._particle.get_in_vertical_boundary_layer()

    @property
    def age(self):
        """ The age of the particle in seconds """
        return self._particle.get_age()

    @property
    def is_alive(self):
        """ Boolean flag indicating whether the particle is alive or dead """
        return self._particle.get_is_alive()


cdef ParticleSmartPtr copy(ParticleSmartPtr particle_smart_ptr):
    """ Create a copy of a ParticleSmartPtr object
    
    This function creates a new copy a ParticleSmartPtr object. In so doing
    new memory is allocated. This memory is automatically freed when the
    ParticleSmartPtr is deleted.
    
    Parameters
    ----------
    particle_smart_ptr : ParticleSmartPtr
        ParticleSmartPtr object.
    
    Returns
    -------
    particle_smart_ptr : ParticleSmartPtr
        An exact copy of the ParticleSmartPtr object passed in.
    """

    return ParticleSmartPtr(particle_smart_ptr=particle_smart_ptr)


cdef to_string(Particle* particle):
    """ Return a string object that describes a particle

    Parameters
    ----------
    particle : Particle C ptr
        Pointer to a particle object

    Returns
    -------
    s : str
        String describing the particle

    """
    cdef vector[string] grids
    cdef vector[int] hosts

    s_base = "Particle properties \n"\
             "------------------- \n"\
             "Particle id = {} \n"\
             "Particle x1 = {} \n"\
             "Particle x2 = {} \n"\
             "Particle x3 = {} \n"\
             "Particle omega interfaces = {} \n"\
             "Particle omega layers = {} \n"\
             "Partilce in vertical boundary layer = {} \n"\
             "Partilce k layer = {} \n"\
             "Partilce k lower layer = {} \n"\
             "Partilce k upper layer = {} \n"\
             "Particle in domain = {} \n"\
             "Particle is beached = {} \n"\
             "Particle age = {} seconds \n".format(particle.get_id(),
                                                   particle.get_x1(),
                                                   particle.get_x2(),
                                                   particle.get_x3(),
                                                   particle.get_omega_interfaces(),
                                                   particle.get_omega_layers(),
                                                   particle.get_in_vertical_boundary_layer(),
                                                   particle.get_k_layer(),
                                                   particle.get_k_lower_layer(),
                                                   particle.get_k_upper_layer(),
                                                   particle.get_in_domain(),
                                                   particle.get_is_beached(),
                                                   particle.get_age())

    # Get host elements
    particle.get_all_host_horizontal_elems(grids, hosts)
    host_elements = {}
    for grid, host in zip(grids, hosts):
        host_elements[grid.decode()] = host

    # Add hosts and phis
    s_hosts = ""
    s_phis = ""
    for key, value in host_elements.items():
        s_hosts += "Host on grid {} = {} \n".format(key, value)

        phis = particle.get_phi(key.encode())
        for i, phi in enumerate(phis):
            s_phis += "Phi {} on grid {} = {} \n".format(i, key, phi)

    s = s_base + s_hosts + s_phis

    return s

