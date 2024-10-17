"""
This module provides a Python friendly API for interacting with
Particle objects. The code for the latter is implemented in C++.
Particle objects include a set of attributes that describe their
state and location within a given domain.
"""

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as preinc

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

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
                  DTYPE_INT_t k_layer=INT_INVALID, bint in_surface_boundary_layer=False,
                  bint in_bottom_boundary_layer=False,
                  DTYPE_INT_t k_lower_layer=INT_INVALID, DTYPE_INT_t k_upper_layer=INT_INVALID,
                  DTYPE_INT_t id=INT_INVALID, DTYPE_INT_t status=0, DTYPE_FLOAT_t age=FLOAT_INVALID,
                  bint is_alive=False, DTYPE_INT_t land_boundary_encounters=0,
                  bint restore_to_fixed_depth=False, DTYPE_FLOAT_t fixed_depth=FLOAT_INVALID,
                  parameters={}, state_variables={}, diagnostic_variables={},
                  boolean_flags={}, ParticleSmartPtr particle_smart_ptr=None):

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
            self._particle.set_in_surface_boundary_layer(in_surface_boundary_layer)
            self._particle.set_in_bottom_boundary_layer(in_bottom_boundary_layer)
            self._particle.set_k_lower_layer(k_lower_layer)
            self._particle.set_k_upper_layer(k_upper_layer)
            self._particle.set_restore_to_fixed_depth(restore_to_fixed_depth)
            self._particle.set_fixed_depth(fixed_depth)
            self._particle.set_age(age)
            self._particle.set_is_alive(is_alive)
            self._particle.set_land_boundary_encounters(land_boundary_encounters)

            # Set local coordinates on all grids
            self.set_all_phis(phis)

            # Add hosts
            self.set_all_host_horizontal_elems(host_elements)

            # Set all parameters
            self.set_all_parameters(parameters)

            # Set all state variables
            self.set_all_state_variables(state_variables)

            # Set all diagnostic variables
            self.set_all_diagnostic_variables(diagnostic_variables)

            # Set all boolean flags
            self.set_all_boolean_flags(boolean_flags)

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

        cdef const vector[DTYPE_FLOAT_t]* _phi = &self._particle.get_phi(grid_name)

        cdef size_t i

        phi = []
        for i in range(_phi.size()):
            phi.append(_phi.at(i))

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

    def set_parameter(self, name, value):
        """ Set generic parameter

        Parameters
        ----------
        name : str
            The name of the parameter.

        value : float
            The parameters value.
        """
        param_name = name.encode() if type(name) == str else name
        self._particle.set_parameter(param_name, value)

    def get_parameter(self, name):
        """ Get parameter

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

        return self._particle.get_parameter(param_name)

    def set_all_parameters(self, parameters):
        """ Set all parameters

        Parameters
        ----------
        parameters : dict
            Dictionary of parameters {name: value}
        """
        self._particle.clear_parameters()
        for name, value in parameters.items():
            self.set_parameter(name, value)

    def get_all_parameters(self):
        """ Get all parameters

        Returns
        -------
        parameters : dict
             Parameters stored in a dictionary.
        """
        cdef vector[string] names
        cdef vector[float] values
        self._particle.get_all_parameters(names, values)

        parameters = {}
        for name, value in zip(names, values):
            parameters[name.decode()] = value

        return parameters

    def set_state_variable(self, name, value):
        """ Set state variable

        Parameters
        ----------
        name : str
            The name of the state variable.

        value : float
            The state variable's value.
        """
        var_name = name.encode() if type(name) == str else name
        self._particle.set_state_variable(var_name, value)

    def get_state_variable(self, name):
        """ Get state variable

        Parameters
        ----------
        name : str
            The name of the state variable.

        Returns
        -------
        value : float
            The value of the state variable.
        """
        var_name = name.encode() if type(name) == str else name

        return self._particle.get_state_variable(var_name)

    def set_all_state_variables(self, state_variables):
        """ Set all state variables

        Parameters
        ----------
        state_variables : dict
            Dictionary of state variables {name: value}
        """
        self._particle.clear_state_variables()
        for name, value in state_variables.items():
            self.set_state_variable(name, value)

    def get_all_state_variables(self):
        """ Get all state variables

        Returns
        -------
        state_variables : dict
             State variables stored in a dictionary.
        """
        cdef vector[string] names
        cdef vector[float] values
        self._particle.get_all_state_variables(names, values)

        state_variables = {}
        for name, value in zip(names, values):
            state_variables[name.decode()] = value

        return state_variables

    def set_diagnostic_variable(self, name, value):
        """ Set diagnostic variable

        Parameters
        ----------
        name : str
            The name of the diagnostic variable.

        value : float
            The diagnostic variable's value.
        """
        var_name = name.encode() if type(name) == str else name
        self._particle.set_diagnostic_variable(var_name, value)

    def get_diagnostic_variable(self, name):
        """ Get diagnostic variable

        Parameters
        ----------
        name : str
            The name of the diagnostic variable.

        Returns
        -------
        value : float
            The value of the diagnostic variable.
        """
        var_name = name.encode() if type(name) == str else name

        return self._particle.get_diagnostic_variable(var_name)

    def set_all_diagnostic_variables(self, diagnostic_variables):
        """ Set all diagnostic variables

        Parameters
        ----------
        diagnostic_variables : dict
            Dictionary of diagnostic variables {name: value}
        """
        self._particle.clear_diagnostic_variables()
        for name, value in diagnostic_variables.items():
            self.set_diagnostic_variable(name, value)

    def get_all_diagnostic_variables(self):
        """ Get all diagnostic variables

        Returns
        -------
        diagnostic_variables : dict
             Diagnostic variables stored in a dictionary.
        """
        cdef vector[string] names
        cdef vector[float] values
        self._particle.get_all_diagnostic_variables(names, values)

        diagnostic_variables = {}
        for name, value in zip(names, values):
            diagnostic_variables[name.decode()] = value

        return diagnostic_variables

    def set_boolean_flag(self, name, value):
        """ Set diagnostic variable

        Parameters
        ----------
        name : str
            The name of the diagnostic variable.

        value : float
            The diagnostic variable's value.
        """
        flag_name = name.encode() if type(name) == str else name
        self._particle.set_boolean_flag(flag_name, value)

    def get_boolean_flag(self, name):
        """ Get diagnostic variable

        Parameters
        ----------
        name : str
            The name of the diagnostic variable.

        Returns
        -------
        value : float
            The value of the diagnostic variable.
        """
        flag_name = name.encode() if type(name) == str else name

        return self._particle.get_boolean_flag(flag_name)

    def set_all_boolean_flags(self, boolean_flags):
        """ Set all boolean flags

        Parameters
        ----------
        boolean_flags : dict
            Dictionary of boolean flags {name: value}
        """
        self._particle.clear_boolean_flags()
        for name, value in boolean_flags.items():
            self.set_boolean_flag(name, value)

    def get_all_boolean_flags(self):
        """ Get all boolean flags

        Returns
        -------
        boolean_flags : dict
             Boolean flags stored in a dictionary.
        """
        cdef vector[string] names
        cdef vector[bool] values
        self._particle.get_all_boolean_flags(names, values)

        boolean_flags = {}
        for name, value in zip(names, values):
            boolean_flags[name.decode()] = value

        return boolean_flags

    @property
    def status(self):
        """ The particle's status (0 - okay; 1 - error) """
        return self._particle.get_status()

    @property
    def x1(self):
        """ The particle's x1-coordinate """
        return self._particle.get_x1()

    @x1.setter
    def x1(self, value):
        self._particle.set_x1(value)

    @property
    def x2(self):
        """ The particle's x2-coordinate """
        return self._particle.get_x2()

    @x2.setter
    def x2(self, value):
        self._particle.set_x2(value)

    @property
    def x3(self):
        """ The particle's x1-coordinate """
        return self._particle.get_x3()

    @x3.setter
    def x3(self, value):
        self._particle.set_x3(value)

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
    def in_surface_boundary_layer(self):
        """ Flag signifying whether or not the particle resides in the surface boundary layer """
        return self._particle.get_in_surface_boundary_layer()

    @property
    def in_bottom_boundary_layer(self):
        """ Flag signifying whether or not the particle resides in the bottom boundary layer """
        return self._particle.get_in_bottom_boundary_layer()

    @property
    def restore_to_fixed_depth(self):
        """ Flag signifying whether a particle's position is restored to a fixed depth """
        return self._particle.get_restore_to_fixed_depth()

    @property
    def fixed_depth(self):
        """ The fixed depth below the surface that particle's are restored to  """
        return self._particle.get_fixed_depth()

    @property
    def age(self):
        """ The age of the particle in seconds """
        return self._particle.get_age()

    @property
    def is_alive(self):
        """ Boolean flag indicating whether the particle is alive or dead """
        return self._particle.get_is_alive()

    @property
    def land_boundary_encounters(self):
        """ The number of times the particle has crossed a land boundary"""
        return self._particle.get_land_boundary_encounters()

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


def to_string_wrapper(ParticleSmartPtr particle):
    """ Python wrapper for to_string
    """
    return to_string(particle.get_ptr())


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
    cdef vector[string] host_grids_c
    cdef vector[int] host_elements_c
    cdef vector[string] phi_grids_c
    cdef vector[vector[DTYPE_FLOAT_t]] phis_c
    cdef size_t i

    s_base = "Particle properties \n"\
             "------------------- \n"\
             "Particle id = {} \n"\
             "Particle x1 = {} \n"\
             "Particle x2 = {} \n"\
             "Particle x3 = {} \n"\
             "Particle omega interfaces = {} \n"\
             "Particle omega layers = {} \n"\
             "Partilce in surface boundary layer = {} \n"\
             "Partilce in bottom boundary layer = {} \n"\
             "Partilce k layer = {} \n"\
             "Partilce k lower layer = {} \n"\
             "Partilce k upper layer = {} \n"\
             "Particle in domain = {} \n"\
             "Particle is beached = {} \n"\
             "Particle is restored to a fixed depth = {} \n"\
             "Particle fixed depth (only used if depth restoring has been activated) = {} \n"\
             "Particle age = {} seconds \n".format(particle.get_id(),
                                                   particle.get_x1(),
                                                   particle.get_x2(),
                                                   particle.get_x3(),
                                                   particle.get_omega_interfaces(),
                                                   particle.get_omega_layers(),
                                                   particle.get_in_surface_boundary_layer(),
                                                   particle.get_in_bottom_boundary_layer(),
                                                   particle.get_k_layer(),
                                                   particle.get_k_lower_layer(),
                                                   particle.get_k_upper_layer(),
                                                   particle.get_in_domain(),
                                                   particle.get_is_beached(),
                                                   particle.get_restore_to_fixed_depth(),
                                                   particle.get_fixed_depth(),
                                                   particle.get_age())

    # Get host elements
    particle.get_all_host_horizontal_elems(host_grids_c, host_elements_c)
    host_elements = {}
    for grid, host in zip(host_grids_c, host_elements_c):
        # Host element
        host_elements[grid.decode()] = host

    # Get phis
    particle.get_all_phis(phi_grids_c, phis_c)
    phis = {}
    for grid, phi in zip(phi_grids_c, phis_c):
        grid_name = grid.decode()
        phis[grid_name] = []
        for i in range(3):
            phis[grid_name].append(phi[i])

    # Add hosts and phis
    s_hosts = ""
    for key, value in host_elements.items():
        s_hosts += "Host on grid {} = {} \n".format(key, value)

    s_phis = ""
    for key, value in phis.items():
        for i in range(3):
            s_phis += "Phi {} on grid {} = {} \n".format(i, key, value[i])

    s = s_base + s_hosts + s_phis

    return s

