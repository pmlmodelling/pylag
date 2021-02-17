"""
Data reader for input data defined on an Arakawa A-grid.

Note
----
arakawa_a_data_reader is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation. However, more
details can be found in `pylag.data_reader`, where a set of python wrappers
have been implemented.
"""


include "constants.pxi"

import logging

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import numpy as np

from cpython cimport bool

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from libcpp.string cimport string
from libcpp.vector cimport vector

# PyLag cython imports
from pylag.particle cimport Particle
from pylag.particle_cpp_wrapper cimport to_string
from pylag.data_reader cimport DataReader
from pylag.unstructured cimport Grid
cimport pylag.interpolation as interp
from pylag.math cimport int_min, float_min
from pylag.math cimport Intersection

# PyLag python imports
from pylag import variable_library
from pylag.unstructured import get_unstructured_grid
from pylag.numerics import get_time_direction


cdef class ArakawaADataReader(DataReader):
    """ DataReader for inputs defined on an Arakawa A-grid
    
    Objects of type ArakawaADataReader are intended to manage all access to
    data objects defined on a Arakawa A-grid, including data describing the
    model grid itself as well as model output variables. Provided are methods
    for searching the model grid for host horizontal elements and for
    interpolating gridded field data to a given point in space and time.
    
    Parameters
    ----------
    config : ConfigParser
        Configuration object.
    
    mediator : Mediator
        Mediator object for managing access to data read from file.


    """
    
    # Configurtion object
    cdef object config

    # Mediator for accessing model data read in from file
    cdef object mediator

    # List of environmental variables to read and save
    cdef object env_var_names

    # Unstructured grid object for performing grid searching etc
    cdef Grid _unstructured_grid

    # The name of the grid
    cdef string _name

    # Grid dimensions
    cdef DTYPE_INT_t _n_longitude, _n_latitude, _n_depth, _n_elems, _n_nodes

    # Flags signifying whether latitude dimension should be clipped
    cdef DTYPE_INT_t _trim_first_latitude, _trim_last_latitude

    # Element connectivity
    cdef DTYPE_INT_t[:,:] _nv
    
    # Element adjacency
    cdef DTYPE_INT_t[:,:] _nbe

    # Node permutation
    cdef DTYPE_INT_t[:] _permutation

    # Minimum nodal x/y values
    cdef DTYPE_FLOAT_t _xmin
    cdef DTYPE_FLOAT_t _ymin
    
    # Reference depth levels, ignoring any changes in sea surface elevation
    cdef DTYPE_FLOAT_t[:] _reference_depth_levels

    # Actual depth levels, accounting for changes in sea surface elevation
    cdef DTYPE_FLOAT_t[:, :] _depth_levels_last
    cdef DTYPE_FLOAT_t[:, :] _depth_levels_next

    # Bathymetry
    cdef DTYPE_FLOAT_t[:] _h

    # Land sea mask on elements (1 - sea point, 0 - land point)
    cdef DTYPE_INT_t[:] _land_sea_mask
    cdef DTYPE_INT_t[:] _land_sea_mask_nodes

    # Full depth mask (1 - sea point, 0 - land point)
    cdef DTYPE_INT_t[:, :] _depth_mask_last
    cdef DTYPE_INT_t[:, :] _depth_mask_next

    # Dictionary of dimension names
    cdef object _dimension_names

    # Dictionary of variable names
    cdef object _variable_names

    # Dictionary containing tuples of variable shapes without time (e.g. {'u': (n_dpeth, n_latitude, n_longitude)})
    cdef object _variable_shapes

    # Dictionaries of variable dimension indices (e.g. {'u': {'depth': 0, 'latitude': 1, 'longitude': 2}})
    cdef object _variable_dimension_indices

    # Sea surface elevation
    cdef DTYPE_FLOAT_t[:] _zeta_last
    cdef DTYPE_FLOAT_t[:] _zeta_next
    
    # u/v/w velocity components
    cdef DTYPE_FLOAT_t[:,:] _u_last
    cdef DTYPE_FLOAT_t[:,:] _u_next
    cdef DTYPE_FLOAT_t[:,:] _v_last
    cdef DTYPE_FLOAT_t[:,:] _v_next
    cdef DTYPE_FLOAT_t[:,:] _w_last
    cdef DTYPE_FLOAT_t[:,:] _w_next
    
    # Vertical eddy diffusivities
    cdef DTYPE_FLOAT_t[:,:] _kh_last
    cdef DTYPE_FLOAT_t[:,:] _kh_next

    # Horizontal eddy viscosities
    cdef DTYPE_FLOAT_t[:,:] _ah_last
    cdef DTYPE_FLOAT_t[:,:] _ah_next

    # Wet/dry status of elements
    cdef DTYPE_INT_t[:] _wet_cells_last
    cdef DTYPE_INT_t[:] _wet_cells_next

    # Sea water potential temperature
    cdef DTYPE_FLOAT_t[:,:] _thetao_last
    cdef DTYPE_FLOAT_t[:,:] _thetao_next

    # Sea water salinity
    cdef DTYPE_FLOAT_t[:,:] _so_last
    cdef DTYPE_FLOAT_t[:,:] _so_next

    # Time direction
    cdef DTYPE_INT_t _time_direction

    # Time array
    cdef DTYPE_FLOAT_t _time_last
    cdef DTYPE_FLOAT_t _time_next

    # Flag for surface only or full 3D transport
    cdef bint _surface_only

    # Flags that identify whether a given variable should be read in
    cdef bint _has_w, _has_Kh, _has_Ah, _has_is_wet,  _has_zeta

    def __init__(self, config, mediator):
        self.config = config
        self.mediator = mediator

        self._name = b'arakawa_a'

        # Time direction
        self._time_direction = <int>get_time_direction(config)

        # 2-D surface only or full 3-D transport
        self._surface_only = self.config.getboolean('SIMULATION', 'surface_only')

        # Setup dimension name mappings
        self._dimension_names = {}
        dim_config_names = {'time': 'time_dim_name', 'depth': 'depth_dim_name', 'latitude': 'latitude_dim_name',
                            'longitude': 'longitude_dim_name'}
        for dim_name, config_name in dim_config_names.items():
            self._dimension_names[dim_name] = self.config.get('OCEAN_CIRCULATION_MODEL', config_name).strip()

        # Setup variable name mappings
        self._variable_names = {}
        var_config_names = {'uo': 'uo_var_name', 'vo': 'vo_var_name', 'wo': 'wo_var_name', 'zos': 'zos_var_name',
                            'Kh': 'Kh_var_name', 'Ah': 'Ah_var_name', 'thetao': 'thetao_var_name',
                            'so': 'so_var_name'}
        for var_name, config_name in var_config_names.items():
            try:
                var = self.config.get('OCEAN_CIRCULATION_MODEL', config_name).strip()
                if var:
                    self._variable_names[var_name] = var
            except (configparser.NoOptionError) as e:
                pass

        # Initialise dictionaries for variable shapes and dimension indices
        self._variable_shapes = {}
        self._variable_dimension_indices = {}

        # Set boolean flags
        self._has_zeta = True if 'zos' in self._variable_names else False
        self._has_w = True if 'wo' in self._variable_names else False
        self._has_Kh = True if 'Kh' in self._variable_names else False
        self._has_Ah = True if 'Ah' in self._variable_names else False

        # Has is wet flag?
        self._has_is_wet = self.config.getboolean("OCEAN_CIRCULATION_MODEL", "has_is_wet")

        # Check to see if any environmental variables are being saved.
        try:
            env_var_names = self.config.get("OUTPUT", "environmental_variables").strip().split(',')
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            env_var_names = []

        self.env_var_names = []
        for env_var_name in env_var_names:
            env_var_name = env_var_name.strip()
            if env_var_name is not None:
                if env_var_name not in self._variable_names.keys():
                    raise ValueError("Received request to track environmental tracer `{}'. However, "\
                            "the corresponding NetCDF variable name was not given. If the variable has been "\
                            "saved within the inputs, please specify its name in the run config.".format(env_var_name))

                if env_var_name not in variable_library.standard_variable_names.keys():
                    raise ValueError("Support for tracking the environmental tracer `{}' does not currently "\
                            "exist within PyLag".format(env_var_name))

                self.env_var_names.append(env_var_name)

        self._read_grid()

        self._read_time_dependent_vars()

    cpdef get_grid_names(self):
        """ Return a list of grid names

        Returns
        -------
         : list [str]
             List of grid names on which which input data are defined.
        """
        return [self._name.decode()]

    cpdef setup_data_access(self, start_datetime, end_datetime):
        """ Set up access to time-dependent variables.
        
        Parameters
        ----------
        start_datetime : Datetime
            Datetime object corresponding to the simulation start time.
        
        end_datetime : Datetime
            Datetime object corresponding to the simulation end time.
        """
        self.mediator.setup_data_access(start_datetime, end_datetime)

        self._read_time_dependent_vars()

    cpdef read_data(self, DTYPE_FLOAT_t time):
        """ Read in time dependent variable data from file?
        
        `time` is used to test if new data should be read in from file. If this
        is the case, arrays containing time-dependent variable data are updated.
        
        Parameters
        ----------
        time : float
            The current time.
        """
        cdef DTYPE_FLOAT_t time_fraction

        time_fraction = interp.get_linear_fraction(time, self._time_last, self._time_next)
        if self._time_direction == 1:
            if time_fraction < 0.0 or time_fraction >= 1.0:
                self.mediator.update_reading_frames(time)
                self._read_time_dependent_vars()
        else:
            if time_fraction <= 0.0 or time_fraction > 1.0:
                self.mediator.update_reading_frames(time)
                self._read_time_dependent_vars()

    cdef DTYPE_INT_t find_host(self, Particle *particle_old,
                               Particle *particle_new) except INT_ERR:
        """ Returns the host horizontal element.
        
        This function first tries to find the new host horizontal element using
        a local search algorithm based on the new point's barycentric
        coordinates. This is relatively fast. However, it can incorrectly flag
        that a particle has left the domain when in-fact it hasn't. For this
        reason, when the local host element search indicates that a particle
        has left the domain, a check is performed based on the particle's
        pathline - if this crosses a known boundary, the particle is deemed
        to have left the domain.

        The function returns a flag that indicates whether or not the particle
        has been found within the domain. If it has, it's host element will 
        have been set appropriately. If not, the the new particle's host
        element will have been set to the last host element the particle passed
        through before exiting the domain.
        
        Conventions
        -----------
        flag = IN_DOMAIN:
            This indicates that the particle was found successfully. Host is the
            index of the new host element.
        
        flag = LAND_BDY_CROSSED:
            This indicates that the particle exited the domain across a land
            boundary. Host is set to the last element the particle passed
            through before exiting the domain.

        flag = OPEN_BDY_CROSSED:
            This indicates that the particle exited the domain across an open
            boundary. Host is set to the last element the particle passed
            through before exiting the domain.

        flag = IN_MASKED_ELEM:
            This indicated the particle is in the domain but the element it is
            in is masked. The flag is only returned by the local and global
            search algorithms.
        
        Parameters
        ----------
        particle_old: *Particle
            The particle at its old position.

        particle_new: *Particle
            The particle at its new position. The host element will be updated.
        
        Returns
        -------
        flag : int
            Integer flag that indicates whether or not the seach was successful.
        """
        cdef DTYPE_INT_t flag, host

        # Save a copy of the current host, then use the old host to start a local search
        host = particle_new.get_host_horizontal_elem(self._name)
        particle_new.set_host_horizontal_elem(self._name, particle_old.get_host_horizontal_elem(self._name))
        flag = self._unstructured_grid.find_host_using_local_search(particle_new)

        if flag == IN_DOMAIN:
            return flag

        # Local search failed to find the particle. Perform check to see if
        # the particle has indeed left the model domain. Reset the host first
        # in order to preserve the original state of particle_new.
        particle_new.set_host_horizontal_elem(self._name, host)
        flag = self._unstructured_grid.find_host_using_particle_tracing(particle_old,
                                                                        particle_new)

        return flag

    cdef DTYPE_INT_t find_host_using_local_search(self, Particle *particle) except INT_ERR:
        """ Returns the host horizontal element through local searching.
        
        This function is a wrapper for the same function implemented in UnstructuredGrid.

        Parameters
        ----------
        particle: *Particle
            The particle.

        Returns
        -------
        flag : int
            Integer flag that indicates whether or not the seach was successful.
        """
        return self._unstructured_grid.find_host_using_local_search(particle)

    cdef DTYPE_INT_t find_host_using_global_search(self, Particle *particle) except INT_ERR:
        """ Returns the host horizontal element through global searching.
        
        This function is a wrapper for the same function implemented in UnstructuredGrid.

        Parameters
        ----------
        particle_old: *Particle
            The particle.
        
        Returns
        -------
        flag : int
            Integer flag that indicates whether or not the seach was successful.
        """
        return self._unstructured_grid.find_host_using_global_search(particle)

    cdef Intersection get_boundary_intersection(self, Particle *particle_old, Particle *particle_new):
        """ Find the boundary intersection point

        This function is a wrapper for the same function implemented in UnstructuredGrid.

        Parameters
        ----------
        particle_old: *Particle
            The particle at its old position.

        particle_new: *Particle
            The particle at its new position.

        Returns
        -------
        intersection: Intersection
            Object describing the boundary intersection.
        """
        return self._unstructured_grid.get_boundary_intersection(particle_old, particle_new)

    cdef set_default_location(self, Particle *particle):
        """ Set default location

        Move the particle to its host element's centroid.
        """
        self._unstructured_grid.set_default_location(particle)

        return

    cdef set_local_coordinates(self, Particle *particle):
        """ Set local coordinates

        This function is a wrapper for the same function implemented in UnstructuredGrid.

        Parameters
        ----------
        particle: *Particle
            Pointer to a Particle struct
        """
        self._unstructured_grid.set_local_coordinates(particle)

        return

    cdef DTYPE_INT_t set_vertical_grid_vars(self, DTYPE_FLOAT_t time,
                                            Particle *particle) except INT_ERR:
        """ Find the host depth layer
        
        Find the depth layer containing x3. The function allows for situations is
        which the particle position lies outside of the specified grid but below
        zeta or above h. In which case, values are generally extrapolated. However,
        while this situation is allowed, it is not prioritised, and when this occurs
        the code run more slowly, as the model will first search the defined grid to
        try and find the particle.
        """
        cdef DTYPE_FLOAT_t depth_upper_level, depth_lower_level

        cdef DTYPE_INT_t mask_upper_level, mask_lower_level

        cdef DTYPE_FLOAT_t h, zeta

        # Host element
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self._name)

        # Time fraction
        cdef DTYPE_FLOAT_t time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        cdef DTYPE_INT_t k

        # Surface only case
        if self._surface_only is True:
            particle.set_in_vertical_boundary_layer(True)
            particle.set_k_layer(0)
            return IN_DOMAIN

        # Loop over all levels to find the host z layer
        for k in xrange(self._n_depth - 1):
            depth_upper_level = self._unstructured_grid.interpolate_in_time_and_space(self._depth_levels_last[k, :],
                                                                                      self._depth_levels_next[k, :],
                                                                                      time_fraction,
                                                                                      particle)

            depth_lower_level = self._unstructured_grid.interpolate_in_time_and_space(self._depth_levels_last[k+1, :],
                                                                                      self._depth_levels_next[k+1, :],
                                                                                      time_fraction,
                                                                                      particle)

            if particle.get_x3() <= depth_upper_level and particle.get_x3() >= depth_lower_level:
                # Host layer found
                particle.set_k_layer(k)

                # Set the sigma level interpolation coefficient
                particle.set_omega_interfaces(interp.get_linear_fraction(particle.get_x3(), depth_lower_level, depth_upper_level))

                # Check to see if any of the nodes on each level are masked
                mask_upper_level = self._interp_mask_status_on_level(host_element, k)
                mask_lower_level = self._interp_mask_status_on_level(host_element, k+1)

                # If the bottom layer is masked, flag the particle as being in the bottom boundary layer.
                if mask_upper_level == 0 and mask_lower_level == 1:
                    particle.set_in_vertical_boundary_layer(True)
                else:
                    particle.set_in_vertical_boundary_layer(False)

                return IN_DOMAIN

        # Allow for the following situations:
        # a) the particle is below zeta but above the highest depth level
        # b) the particle is above h but below the lowest depth level.
        # If this has happened, flag the particle as being in the vertical boundary layer, meaning all
        # variables will be extrapolated from the below/above depth level.
        h = self.get_zmin(time, particle)
        zeta = self.get_zmax(time, particle)

        # Particle is below zeta but above the top depth level
        k = 0
        depth_upper_level = self._unstructured_grid.interpolate_in_time_and_space(self._depth_levels_last[k, :],
                                                                                  self._depth_levels_next[k, :],
                                                                                  time_fraction,
                                                                                  particle)
        if particle.get_x3() <= zeta and particle.get_x3() >= depth_lower_level:
            mask_upper_level = self._interp_mask_status_on_level(host_element, k)
            if mask_upper_level == 0:
                particle.set_k_layer(k)
                particle.set_in_vertical_boundary_layer(True)

                return IN_DOMAIN
            else:
                raise RuntimeError('The particle sits below zeta and above the top level, which is masked. Why should it be masked?')

        # Particle is above h but below the lowest depth level
        k = self._n_depth - 1
        depth_upper_level = self._unstructured_grid.interpolate_in_time_and_space(self._depth_levels_last[k, :],
                                                                                  self._depth_levels_next[k, :],
                                                                                  time_fraction,
                                                                                  particle)
        if particle.get_x3() >= h and particle.get_x3() <= depth_upper_level:
            mask_upper_level = self._interp_mask_status_on_level(host_element, k)
            if mask_upper_level == 0:
                particle.set_k_layer(k)
                particle.set_in_vertical_boundary_layer(True)

                return IN_DOMAIN
            else:
                raise RuntimeError('The particle sits above h but below the bottom level, which is masked. Why should it be masked?')

        # Particle is outside the vertical grid
        return BDY_ERROR

    cpdef DTYPE_FLOAT_t get_xmin(self) except FLOAT_ERR:
        """ Get minimum x-value for the domain

        Returns
        -------
         : float
             The minimum value of `x` across the grid.
        """
        return self._xmin

    cpdef DTYPE_FLOAT_t get_ymin(self) except FLOAT_ERR:
        """ Get minimum y-value for the domain

        Returns
        -------
         : float
             The minimum value of `y` across the grid.
        """
        return self._ymin

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        """ Returns the bottom depth in cartesian coordinates

        h is defined at element nodes. Linear interpolation in space is used
        to compute h(x,y). NB the negative of h (which is +ve downwards) is
        returned.

        Parameters
        ----------
        time : float
            Time.

        particle: *Particle
            Pointer to a Particle object.

        Returns
        -------
        zmin : float
            The bottom depth.
        """
        cdef DTYPE_FLOAT_t h # Bathymetry at (x1, x2)

        if self._surface_only:
            return 0.0

        h = self._unstructured_grid.interpolate_in_space(self._h, particle)

        return -h

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        """ Returns the sea surface height in cartesian coordinates

        zeta is defined at element nodes. Interpolation proceeds through linear
        interpolation in time followed by interpolation in space.

        Parameters
        ----------
        time : float
            Time.

        particle: *Particle
            Pointer to a Particle object.
        
        Returns
        -------
        zmax : float
            Sea surface elevation.
        """
        cdef DTYPE_FLOAT_t time_fraction # Time interpolation coefficient
        cdef DTYPE_FLOAT_t zeta # Sea surface elevation at (t, x1, x2)

        if self._surface_only == True or self._has_zeta == False:
            return 0.0

        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        zeta = self._unstructured_grid.interpolate_in_time_and_space(self._zeta_last,
                                                                     self._zeta_next,
                                                                     time_fraction,
                                                                     particle)

        return zeta

    cdef get_velocity(self, DTYPE_FLOAT_t time, Particle* particle,
            DTYPE_FLOAT_t vel[3]):
        """ Returns the velocity u(t,x,y,z) through linear interpolation
        
        Returns the velocity u(t,x,y,z) through interpolation for a particle.

        Parameters
        ----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.

        Return
        ------
        vel : C array, float
            u/v/w velocity components stored in a C array.           
        """
        vel[0] = self._get_variable(self._u_last, self._u_next, time, particle)
        vel[1] = self._get_variable(self._v_last, self._v_next, time, particle)
        if not self._surface_only and self._has_w:
            vel[2] = self._get_variable(self._w_last, self._w_next, time, particle)
        else:
            vel[2] = 0.0

        return

    cdef DTYPE_FLOAT_t get_environmental_variable(self, var_name,
            DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        """ Returns the value of the given environmental variable through linear interpolation

        Support for extracting the following environmental variables has been implemented:

        thetao - Sea water potential temperature

        so - Sea water salinty

        Parameters
        ----------
        var_name : str
            The name of the variable. See above for a list of supported options.

        time : float
            Time at which to interpolate.

        particle: *Particle
            Pointer to a Particle object.

        Returns:
        --------
        var : float
            The interpolated value of the variable at the specified point in time and space.
        """
        cdef DTYPE_FLOAT_t var # Environmental variable at (t, x1, x2, x3)

        if var_name in self.env_var_names:
            if var_name == 'thetao':
                var = self._get_variable(self._thetao_last, self._thetao_next, time, particle)
            elif var_name == 'so':
                var = self._get_variable(self._so_last, self._so_next, time, particle)
            return var
        else:
            raise ValueError("Invalid variable name `{}'".format(var_name))

    cdef get_horizontal_eddy_viscosity(self, DTYPE_FLOAT_t time,
            Particle* particle):
        """ Returns the horizontal eddy viscosity through linear interpolation

        Parameters
        ----------
        time : float
            Time at which to interpolate.

        particle: *Particle
            Pointer to a Particle object.

        Returns
        -------
        viscofh : float
            The interpolated value of the horizontal eddy viscosity at the specified point in time and space.
        """
        cdef DTYPE_FLOAT_t var  # ah at (t, x1, x2, x3)

        var = self._get_variable(self._ah_last, self._ah_next, time, particle)

        return var

    cdef get_horizontal_eddy_viscosity_derivative(self, DTYPE_FLOAT_t time,
            Particle* particle, DTYPE_FLOAT_t Ah_prime[2]):
        """ Returns the gradient in the horizontal eddy viscosity

        Parameters
        ----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.

        Ah_prime : C array, float
            dAh_dx and dH_dy components stored in a C array of length two.  

        TODO
        ----
        1) Test this.

        References
        ----------
        Lynch, D. R. et al (2014). Particles in the coastal ocean: theory and
        applications. Cambridge: Cambridge University Press.
        doi.org/10.1017/CBO9781107449336
        """
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier

        # Variables used in interpolation in time
        cdef DTYPE_FLOAT_t time_fraction

        # Particle k_layer
        cdef DTYPE_INT_t k_layer = particle.get_k_layer()

        # Host element
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self._name)

        # Gradients in phi
        cdef vector[DTYPE_FLOAT_t] dphi_dx = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)
        cdef vector[DTYPE_FLOAT_t] dphi_dy = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)

        # Intermediate arrays - ah
        cdef vector[DTYPE_FLOAT_t] ah_tri_t_last_level_1 = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)
        cdef vector[DTYPE_FLOAT_t] ah_tri_t_next_level_1 = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)
        cdef vector[DTYPE_FLOAT_t] ah_tri_t_last_level_2 = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)
        cdef vector[DTYPE_FLOAT_t] ah_tri_t_next_level_2 = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)
        cdef vector[DTYPE_FLOAT_t] ah_tri_level_1 = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)
        cdef vector[DTYPE_FLOAT_t] ah_tri_level_2 = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)

        # Gradients on lower and upper bounding sigma layers
        cdef DTYPE_FLOAT_t dah_dx_level_1
        cdef DTYPE_FLOAT_t dah_dy_level_1
        cdef DTYPE_FLOAT_t dah_dx_level_2
        cdef DTYPE_FLOAT_t dah_dy_level_2

        # Gradient
        cdef DTYPE_FLOAT_t dah_dx
        cdef DTYPE_FLOAT_t dah_dy

        # Time fraction
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        # Get gradient in phi
        self._unstructured_grid.get_grad_phi(host_element, dphi_dx, dphi_dy)

        # No vertical interpolation for particles near to the bottom,
        if particle.get_in_vertical_boundary_layer() is True:
            # Extract ah near to the boundary
            for i in xrange(N_VERTICES):
                vertex = self._nv[i, host_element]
                ah_tri_t_last_level_1[i] = self._ah_last[k_layer, vertex]
                ah_tri_t_next_level_1[i] = self._ah_next[k_layer, vertex]

            # Interpolate in time
            for i in xrange(N_VERTICES):
                ah_tri_level_1[i] = interp.linear_interp(time_fraction,
                                            ah_tri_t_last_level_1[i],
                                            ah_tri_t_next_level_1[i])

            # Interpolate d{}/dx and d{}/dy within the host element
            Ah_prime[0] = interp.interpolate_within_element(ah_tri_level_1, dphi_dx)
            Ah_prime[1] = interp.interpolate_within_element(ah_tri_level_1, dphi_dy)
            return
        else:
            # Extract ah on the lower and upper bounding depth levels
            for i in xrange(N_VERTICES):
                vertex = self._nv[i,host_element]
                ah_tri_t_last_level_1[i] = self._ah_last[k_layer+1, vertex]
                ah_tri_t_next_level_1[i] = self._ah_next[k_layer+1, vertex]
                ah_tri_t_last_level_2[i] = self._ah_last[k_layer, vertex]
                ah_tri_t_next_level_2[i] = self._ah_next[k_layer, vertex]

            # Interpolate in time
            for i in xrange(N_VERTICES):
                ah_tri_level_1[i] = interp.linear_interp(time_fraction,
                                            ah_tri_t_last_level_1[i],
                                            ah_tri_t_next_level_1[i])
                ah_tri_level_2[i] = interp.linear_interp(time_fraction,
                                            ah_tri_t_last_level_2[i],
                                            ah_tri_t_next_level_2[i])

            # Interpolate d{}/dx and d{}/dy within the host element on the upper
            # and lower bounding depth levels
            dah_dx_level_1 = interp.interpolate_within_element(ah_tri_level_1, dphi_dx)
            dah_dy_level_1 = interp.interpolate_within_element(ah_tri_level_1, dphi_dy)
            dah_dx_level_2 = interp.interpolate_within_element(ah_tri_level_2, dphi_dx)
            dah_dy_level_2 = interp.interpolate_within_element(ah_tri_level_2, dphi_dy)

            # Interpolate d{}/dx and d{}/dy between bounding depth levels and
            # save in the array Ah_prime
            Ah_prime[0] = interp.linear_interp(particle.get_omega_interfaces(), dah_dx_level_1, dah_dx_level_2)
            Ah_prime[1] = interp.linear_interp(particle.get_omega_interfaces(), dah_dy_level_1, dah_dy_level_2)

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            Particle* particle) except FLOAT_ERR:
        """ Returns the vertical eddy diffusivity through linear interpolation.
        
        Parameters
        ----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.
        
        Returns
        -------
        kh : float
            The vertical eddy diffusivity.        
        
        """
        cdef DTYPE_FLOAT_t var  # kh at (t, x1, x2, x3)

        if not self._surface_only:
            var = self._get_variable(self._kh_last, self._kh_next, time, particle)
        else:
            var = 0.0

        return var

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self,
            DTYPE_FLOAT_t time, Particle* particle) except FLOAT_ERR:
        """ Returns the gradient in the vertical eddy diffusivity.
        
        Return a numerical approximation of the gradient in the vertical eddy 
        diffusivity at (t,x,y,z) using central differencing. First, the
        diffusivity is computed on the depth levels bounding the particle.
        Central differencing is then used to compute the gradient in the
        diffusivity on these levels. Finally, the gradient in the diffusivity
        is interpolated to the particle's exact position. This algorithm
        mirrors that used in GOTMDataReader, which is why it has been implemented
        here. However, in contrast to GOTMDataReader, which calculates the
        gradient in the diffusivity at all levels once each simulation time step,
        resulting in significant time savings, this function is executed once
        for each particle. It is thus quite costly! To make things worse, the 
        code, as implemented here, is highly repetitive, and no doubt efficiency
        savings could be found. 

        TODO
        ----
        1) Can this be done using a specialised interpolating object? Would need to
        compute depths and kh at all levels, then interpolate between then. Would
        need to give some thought to how to do this efficiently.
        2) Test this.

        Parameters
        ----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.
        
        Returns
        -------
        k_prime : float
            Gradient in the vertical eddy diffusivity field.
        """
        cdef DTYPE_FLOAT_t kh_0, kh_1, kh_2, kh_3
        cdef DTYPE_FLOAT_t z_0, z_1, z_2, z_3
        cdef DTYPE_FLOAT_t dkh_lower_level, dkh_upper_level

        # Particle k_layer
        cdef DTYPE_INT_t k_layer = particle.get_k_layer()

        if self._surface_only:
            return 0.0

        raise NotImplementedError('Implementation is under development')

#        if particle.get_in_vertical_boundary_layer() == True:
#            kh_0 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, k_layer-1)
#            z_0 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer-1)
#
#            kh_1 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, k_layer)
#            z_1 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer)
#
#            dkh_upper_level = (kh_0 - kh_1) / (z_0 - z_1)
#
#            return dkh_upper_level
#
#        if k_layer == 0:
#            kh_0 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, k_layer)
#            z_0 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer)
#
#            kh_1 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, k_layer+1)
#            z_1 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer+1)
#
#            kh_2 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, k_layer+2)
#            z_2 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer+2)
#
#            dkh_lower_level = (kh_0 - kh_2) / (z_0 - z_2)
#            dkh_upper_level = (kh_0 - kh_1) / (z_0 - z_1)
#
#        elif k_layer == self._n_depth - 2:
#            kh_0 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, k_layer-1)
#            z_0 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer-1)
#
#            kh_1 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, k_layer)
#            z_1 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer)
#
#            kh_2 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, k_layer+1)
#            z_2 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer+1)
#
#            dkh_lower_level = (kh_1 - kh_2) / (z_1 - z_2)
#            dkh_upper_level = (kh_0 - kh_2) / (z_0 - z_2)
#
#        else:
#            kh_0 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, k_layer-1)
#            z_0 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer-1)
#
#            kh_1 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, k_layer)
#            z_1 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer)
#
#            kh_2 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, k_layer+1)
#            z_2 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer+1)
#
#            kh_3 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, k_layer+2)
#            z_3 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer+2)
#
#            dkh_lower_level = (kh_1 - kh_3) / (z_1 - z_3)
#            dkh_upper_level = (kh_0 - kh_2) / (z_0 - z_2)
#
#        return interp.linear_interp(particle.get_omega_interfaces(), dkh_lower_level, dkh_upper_level)

    cdef DTYPE_INT_t is_wet(self, DTYPE_FLOAT_t time, Particle *particle) except INT_ERR:
        """ Return an integer indicating whether `host' is wet or dry
        
        The function returns 1 if `host' is wet at time `time' and 
        0 if `host' is dry.
        
        The wet-dry distinction reflects two discrete states - either the
        element is wet, or it is dry. This raises the question of how to deal
        with intermediate times, such that td < t < tw where
        t is the current model time, and td and tw are conescutive input time
        points between which the host element switches from being dry to being
        wet. The approach taken is conservative, and involves flagging the
        element as being dry if either one or both of the input time points
        bounding the current model time indicate that the element is dry. In this
        simple procedure, the `time' parameter is actually unused.
        
        NB - just because an element is flagged as being dry does not mean
        that particles are necessarily frozen. Clients can still try to advect
        particles within such elements, and the interpolated velocity field may
        yield non-zero values, depending on the state of the host and
        surrounding elements in the given time window.
        
        Parameters
        ----------
        time : float
            Time (unused)

        host : int
            Integer that identifies the host element in question
        """
        cdef DTYPE_FLOAT_t zmin_last, zmax_last
        cdef DTYPE_FLOAT_t zmin_mext, zmax_next
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self._name)

        # In surface only transport case don't account for wetting and drying
        if self._surface_only:
            return 1

        zmin_last = self.get_zmin(self._time_last, particle)
        zmax_last = self.get_zmax(self._time_last, particle)
        zmin_next = self.get_zmin(self._time_next, particle)
        zmax_next = self.get_zmax(self._time_next, particle)

        if self._wet_cells_last[host_element] == 0 or self._wet_cells_next[host_element] == 0:
            return 0

        if zmax_last < zmin_last or zmax_next < zmin_next:
            return 0

        return 1

    cdef DTYPE_FLOAT_t _get_variable(self, DTYPE_FLOAT_t[:, :] var_last, DTYPE_FLOAT_t[:, :] var_next,
            DTYPE_FLOAT_t time, Particle* particle) except FLOAT_ERR:
        """ Returns the value of the variable through linear interpolation

        Private method for interpolating fields specified at element nodes on depth levels.
        For particle at depths above h and above a lower level with masked nodes, extrapolation
        is used.

        Parameters
        ----------
        var_last : 2D MemoryView
            Array of variable values at the last time index.

        var_next : 2D MemoryView
            Array of variable values at the next time index.

        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object. 
        
        Returns
        -------
        var : float
            The interpolated value of the variable at the specified point in time and space.
        """
        # Interpolated values on lower and upper bounding depth levels
        cdef DTYPE_FLOAT_t var_level_1
        cdef DTYPE_FLOAT_t var_level_2

        # Particle k_layer
        cdef DTYPE_INT_t k_layer = particle.get_k_layer()

        cdef DTYPE_FLOAT_t time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        # No vertical interpolation for particles near to the bottom
        if particle.get_in_vertical_boundary_layer() is True:
            return self._unstructured_grid.interpolate_in_time_and_space(var_last[k_layer, :],
                                                                         var_next[k_layer, :],
                                                                         time_fraction,
                                                                         particle)
        else:
            var_level_1 = self._unstructured_grid.interpolate_in_time_and_space(var_last[k_layer+1, :],
                                                                                var_next[k_layer+1, :],
                                                                                time_fraction,
                                                                                particle)

            var_level_2 = self._unstructured_grid.interpolate_in_time_and_space(var_last[k_layer, :],
                                                                                var_next[k_layer, :],
                                                                                time_fraction,
                                                                                particle)

            return interp.linear_interp(particle.get_omega_interfaces(), var_level_1, var_level_2)

    def _read_grid(self):
        """ Set grid and coordinate variables.
        
        All communications go via the mediator in order to guarantee support for
        both serial and parallel simulations.
        
        Parameters
        ----------
        N/A
        
        Returns
        -------
        N/A
        """
        # Read in the grid's dimensions
        self._n_longitude = self.mediator.get_dimension_variable('longitude')
        self._n_latitude = self.mediator.get_dimension_variable('latitude')
        self._n_nodes = self.mediator.get_dimension_variable('node')
        self._n_elems = self.mediator.get_dimension_variable('element')

        if not self._surface_only:
            self._n_depth = self.mediator.get_dimension_variable('depth')
        else:
            self._n_depth = 1

        # Read in flags signifying whether time dependent variable arrays should be trimmed
        # in order to eliminate gridded data at -90 deg. or 90 deg. N.
        self._trim_first_latitude = self.mediator.get_grid_variable('trim_first_latitude', (1), DTYPE_INT)
        self._trim_last_latitude = self.mediator.get_grid_variable('trim_last_latitude', (1), DTYPE_INT)

        # Grid connectivity/adjacency
        self._nv = self.mediator.get_grid_variable('nv', (3, self._n_elems), DTYPE_INT)
        self._nbe = self.mediator.get_grid_variable('nbe', (3, self._n_elems), DTYPE_INT)

        # Node permutation
        self._permutation = self.mediator.get_grid_variable('permutation', (self._n_nodes), DTYPE_INT)

        # Raw grid x/y or lat/lon coordinates
        coordinate_system = self.config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system").strip().lower()

        if coordinate_system == "geographic":
            x = self.mediator.get_grid_variable('longitude', (self._n_nodes), DTYPE_FLOAT)
            y = self.mediator.get_grid_variable('latitude', (self._n_nodes), DTYPE_FLOAT)
            xc = self.mediator.get_grid_variable('longitude_c', (self._n_elems), DTYPE_FLOAT)
            yc = self.mediator.get_grid_variable('latitude_c', (self._n_elems), DTYPE_FLOAT)

            # Don't apply offsets in geographic case - set them to 0.0!
            self._xmin = 0.0
            self._ymin = 0.0
        else:
            raise ValueError("Unsupported model coordinate system `{}'".format(coordinate_system))

        # Apply offsets
        x = x - self._xmin
        y = y - self._ymin
        xc = xc - self._xmin
        yc = yc - self._ymin

        # Land sea mask
        self._land_sea_mask = self.mediator.get_grid_variable('mask', (self._n_elems), DTYPE_INT)
        self._land_sea_mask_nodes = self.mediator.get_grid_variable('mask_nodes', (self._n_nodes), DTYPE_INT)

        # Initialise unstructured grid
        self._unstructured_grid = get_unstructured_grid(self.config, self._name, self._n_nodes, self._n_elems,
                                                        self._nv, self._nbe, x, y, xc, yc, self._land_sea_mask,
                                                        self._land_sea_mask_nodes)

        # Read in depth vars if doing a 3D run
        if not self._surface_only:
            # Depth levels at nodal coordinates. Assumes and requires that depth is positive down. The -1 multiplier
            # flips this so that depth is positive up from the zero geoid.
            self._reference_depth_levels = -1.0 * self.mediator.get_grid_variable('depth', (self._n_depth), DTYPE_FLOAT)

            # Bathymetry
            self._h = self.mediator.get_grid_variable('h', (self._n_nodes), DTYPE_FLOAT)

            # Initialise depth level arrays (but don't fill for now)
            self._depth_levels_last = np.empty((self._n_depth, self._n_nodes), dtype=DTYPE_FLOAT)
            self._depth_levels_next = np.empty((self._n_depth, self._n_nodes), dtype=DTYPE_FLOAT)


        # Initialise is wet arrays (but don't fill for now)
        self._wet_cells_last = np.empty((self._n_elems), dtype=DTYPE_INT)
        self._wet_cells_next = np.empty((self._n_elems), dtype=DTYPE_INT)

        # Add zeta to shape and dimension indices dictionaries
        if self._has_zeta:
            self._variable_shapes['zos'] = self.mediator.get_variable_shape(self._variable_names['zos'])[1:]
            dimensions = self.mediator.get_variable_dimensions(self._variable_names['zos'])[1:]
            self._variable_dimension_indices['zos'] = {'latitude': dimensions.index(self._dimension_names['latitude']),
                                                       'longitude': dimensions.index(self._dimension_names['longitude'])}

        # Add 3D vars to shape and dimension indices dictionaries
        var_names = ['uo', 'vo']
        if self._has_w: var_names.append('wo')
        if self._has_Kh: var_names.append('Kh')
        if self._has_Ah: var_names.append('Ah')
        if 'thetao' in self.env_var_names: var_names.append('thetao')
        if 'so' in self.env_var_names: var_names.append('so')
        for var_name in var_names:
            self._variable_shapes[var_name] = self.mediator.get_variable_shape(self._variable_names[var_name])[1:]
            dimensions = self.mediator.get_variable_dimensions(self._variable_names[var_name])[1:]
            self._variable_dimension_indices[var_name] = {'depth': dimensions.index(self._dimension_names['depth']),
                                                          'latitude': dimensions.index(self._dimension_names['latitude']),
                                                          'longitude': dimensions.index(self._dimension_names['longitude'])}

    cdef _read_time_dependent_vars(self):
        """ Update time variables and memory views for data fields.
        
        For each time-dependent variable needed by PyLag two references
        are stored. These correspond to the last and next time points at which
        data was saved. Together these bound PyLag's current time point.
        
        All communications go via the mediator in order to guarantee support for
        both serial and parallel simulations.
        
        Parameters
        ----------
        N/A
        
        Returns
        -------
        N/A
        """
        cdef DTYPE_INT_t i, j, k
        cdef DTYPE_INT_t node
        cdef DTYPE_INT_t is_wet_last, is_wet_next

        # Update time references
        self._time_last = self.mediator.get_time_at_last_time_index()
        self._time_next = self.mediator.get_time_at_next_time_index()

        # Update memory views for zeta
        if self._has_zeta:
            zeta_var_name = self._variable_names['zos']

            # Zeta at last time step
            zeta_last = self.mediator.get_time_dependent_variable_at_last_time_index(zeta_var_name,
                    self._variable_shapes['zos'], DTYPE_FLOAT)
            self._zeta_last = self._reshape_var(zeta_last, self._variable_dimension_indices['zos'])

            # Zeta at next time step
            zeta_next = self.mediator.get_time_dependent_variable_at_next_time_index(zeta_var_name,
                    self._variable_shapes['zos'], DTYPE_FLOAT)
            self._zeta_next = self._reshape_var(zeta_next, self._variable_dimension_indices['zos'])

        # Update memory views for u
        u_var_name = self._variable_names['uo']
        u_last = self.mediator.get_time_dependent_variable_at_last_time_index(u_var_name,
                self._variable_shapes['uo'], DTYPE_FLOAT)
        self._u_last = self._reshape_var(u_last, self._variable_dimension_indices['uo'])

        u_next = self.mediator.get_time_dependent_variable_at_next_time_index(u_var_name,
                self._variable_shapes['uo'], DTYPE_FLOAT)
        self._u_next = self._reshape_var(u_next, self._variable_dimension_indices['uo'])

        # Update memory views for v
        v_var_name = self._variable_names['vo']
        v_last = self.mediator.get_time_dependent_variable_at_last_time_index(v_var_name,
                self._variable_shapes['vo'], DTYPE_FLOAT)
        self._v_last = self._reshape_var(v_last, self._variable_dimension_indices['vo'])

        v_next = self.mediator.get_time_dependent_variable_at_next_time_index(v_var_name,
                self._variable_shapes['vo'], DTYPE_FLOAT)
        self._v_next = self._reshape_var(v_next, self._variable_dimension_indices['vo'])

        # Update memory views for w
        if self._has_w:
            w_var_name = self._variable_names['wo']
            w_last = self.mediator.get_time_dependent_variable_at_last_time_index(w_var_name,
                    self._variable_shapes['wo'], DTYPE_FLOAT)
            self._w_last = self._reshape_var(w_last, self._variable_dimension_indices['wo'])

            w_next = self.mediator.get_time_dependent_variable_at_next_time_index(w_var_name,
                    self._variable_shapes['wo'], DTYPE_FLOAT)
            self._w_next = self._reshape_var(w_next, self._variable_dimension_indices['wo'])

        # Update depth mask
        if not self._surface_only:
            depth_mask_last = self.mediator.get_mask_at_last_time_index(u_var_name,
                    self._variable_shapes['uo'])
            self._depth_mask_last = self._reshape_var(depth_mask_last, self._variable_dimension_indices['uo'])

            depth_mask_next = self.mediator.get_mask_at_next_time_index(u_var_name,
                    self._variable_shapes['uo'])
            self._depth_mask_next = self._reshape_var(depth_mask_next, self._variable_dimension_indices['uo'])

            # Compute actual depth levels using reference values and zeta
            for k in xrange(self._n_depth):
                for i in xrange(self._n_nodes):
                    self._depth_levels_last[k, i] = self._reference_depth_levels[k] + self._zeta_last[i]
                    self._depth_levels_next[k, i] = self._reference_depth_levels[k] + self._zeta_next[i]

        # Update memory views for kh
        if self._has_Kh:
            kh_var_name = self._variable_names['Kh']
            kh_last = self.mediator.get_time_dependent_variable_at_last_time_index(kh_var_name,
                    self._variable_shapes['Kh'], DTYPE_FLOAT)
            self._kh_last = self._reshape_var(kh_last, self._variable_dimension_indices['Kh'])

            kh_next = self.mediator.get_time_dependent_variable_at_next_time_index(kh_var_name,
                    self._variable_shapes['Kh'], DTYPE_FLOAT)
            self._kh_next = self._reshape_var(kh_next, self._variable_dimension_indices['Kh'])

        # Update memory views for Ah
        if self._has_Ah:
            ah_var_name = self._variable_names['Ah']
            ah_last = self.mediator.get_time_dependent_variable_at_last_time_index(ah_var_name,
                    self._variable_shapes['Ah'], DTYPE_FLOAT)
            self._ah_last = self._reshape_var(ah_last, self._variable_dimension_indices['Ah'])

            ah_next = self.mediator.get_time_dependent_variable_at_next_time_index(ah_var_name,
                    self._variable_shapes['Ah'], DTYPE_FLOAT)
            self._ah_next = self._reshape_var(ah_next, self._variable_dimension_indices['Ah'])

        # Set is wet status
        # NB the status of cells is inferred from the depth mask and the land-sea element mask. If a surface cell is
        # masked but it is not a land cell, then it is assumed to be dry.
        if not self._surface_only:
            for i in xrange(self._n_elems):
                if self._land_sea_mask[i] == 0:
                    is_wet_last = 1
                    is_wet_next = 1
                    for j in xrange(3):
                        node = self._nv[j, i]
                        if self._depth_mask_last[0, node] == 1:
                            is_wet_last = 0
                        if self._depth_mask_next[0, node] == 1:
                            is_wet_next = 0
                    self._wet_cells_last[i] = is_wet_last
                    self._wet_cells_next[i] = is_wet_next

        # Read in data as requested
        if 'thetao' in self.env_var_names:
            var_name = self._variable_names['thetao']
            thetao_next = self.mediator.get_time_dependent_variable_at_next_time_index(var_name,
                    self._variable_shapes['thetao'], DTYPE_FLOAT)
            self._thetao_next = self._reshape_var(thetao_next, self._variable_dimension_indices['thetao'])

            thetao_last = self.mediator.get_time_dependent_variable_at_last_time_index(var_name,
                    self._variable_shapes['thetao'], DTYPE_FLOAT)
            self._thetao_last = self._reshape_var(thetao_last, self._variable_dimension_indices['thetao'])

        if 'so' in self.env_var_names:
            var_name = self._variable_names['so']
            so_next = self.mediator.get_time_dependent_variable_at_next_time_index(var_name,
                    self._variable_shapes['so'], DTYPE_FLOAT)
            self._so_next = self._reshape_var(so_next, self._variable_dimension_indices['so'])

            so_last = self.mediator.get_time_dependent_variable_at_last_time_index(var_name,
                    self._variable_shapes['so'], DTYPE_FLOAT)
            self._so_last = self._reshape_var(so_last, self._variable_dimension_indices['so'])

        return

    def _reshape_var(self, var, dimension_indices):
        """ Reshape variable for PyLag

        Variables with the following dimensions are supported:

        2D - [lat, lon] in any order

        3D - [depth, lat, lon] in any order

        Latitude trimming is applied as required.

        Parameters
        ----------
        var : NDArray
            The variable to sort

        dimension_indices : dict
            Dictionary of dimension indices

        Returns
        -------
        var_reshaped : NDArray
            Reshaped variable
        """
        n_dimensions = len(var.shape)

        if n_dimensions == 2:
            lat_index = dimension_indices['latitude']
            lon_index = dimension_indices['longitude']

            # Shift axes to give [x, y]
            var = np.moveaxis(var, lon_index, 0)

            # Trim latitudes
            if self._trim_first_latitude == 1:
                var = var[:, 1:]
            if self._trim_last_latitude == 1:
                var = var[:, :-1]

            return var.reshape(np.prod(var.shape), order='C')[self._permutation]

        elif n_dimensions == 3:
            depth_index = dimension_indices['depth']
            lat_index = dimension_indices['latitude']
            lon_index = dimension_indices['longitude']

            # Shift axes to give [z, x, y]
            var = np.moveaxis(var, depth_index, 0)

            # Update lat/lon indices if needed
            if depth_index > lat_index:
                lat_index += 1
            if depth_index > lon_index:
                lon_index += 1

            var = np.moveaxis(var, lon_index, 1)

            # Trim latitudes
            if self._trim_first_latitude == 1:
                var = var[:, :, 1:]
            if self._trim_last_latitude == 1:
                var = var[:, :, :-1]

            return var.reshape(var.shape[0], np.prod(var.shape[1:]), order='C')[:, self._permutation]
        else:
            raise ValueError('Unsupported number of dimensions {}.'.format(n_dimensions))

    cdef DTYPE_INT_t _interp_mask_status_on_level(self,
            DTYPE_INT_t host, DTYPE_INT_t kidx) except INT_ERR:
        """ Return the masked status of the given depth level
 
        Parameters
        ----------
        phi : c array, float
            Array of length three giving the barycentric coordinates at which 
            to interpolate.
            
        host : int
            Host element index.

        kidx : int
            Sigma layer on which to interpolate.

        Returns
        -------
        mask : int
            Masked status (1 is masked, 0 not masked).
        """
        cdef int vertex # Vertex identifier
        cdef DTYPE_INT_t mask_status_last, mask_status_next
        cdef DTYPE_INT_t mask
        cdef DTYPE_INT_t i

        mask = 0
        for i in xrange(N_VERTICES):
            vertex = self._nv[i,host]
            mask_status_last = self._depth_mask_last[kidx, vertex]
            mask_status_next = self._depth_mask_next[kidx, vertex]

            if mask_status_last == 1 or mask_status_next == 1:
                mask = 1
                break

        return mask

