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
from libc.math cimport sqrt

# PyLag cython imports
from pylag.parameters cimport deg_to_radians
from pylag.particle cimport Particle
from pylag.particle_cpp_wrapper cimport to_string
from pylag.data_reader cimport DataReader
from pylag.unstructured cimport Grid
cimport pylag.interpolation as interp
from pylag.math cimport int_min, float_min

# PyLag python imports
from pylag import variable_library
from pylag.exceptions import PyLagValueError
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
    cdef DTYPE_INT_t[:,::1] _nv
    
    # Element adjacency
    cdef DTYPE_INT_t[:,::1] _nbe

    # Node permutation
    cdef DTYPE_INT_t[::1] _permutation

    # Minimum nodal x/y values
    cdef DTYPE_FLOAT_t _xmin
    cdef DTYPE_FLOAT_t _ymin
    
    # Reference depth levels, ignoring any changes in sea surface elevation
    cdef DTYPE_FLOAT_t[::1] _reference_depth_levels

    # Actual depth levels, accounting for changes in sea surface elevation
    cdef DTYPE_FLOAT_t[:, ::1] _depth_levels_last
    cdef DTYPE_FLOAT_t[:, ::1] _depth_levels_next

    # Bathymetry
    cdef DTYPE_FLOAT_t[::1] _h

    # Land sea mask on elements (1 - sea point, 0 - land point)
    cdef DTYPE_INT_t[::1] _land_sea_mask_c
    cdef DTYPE_INT_t[::1] _land_sea_mask

    # Full depth mask (1 - sea point, 0 - land point)
    cdef DTYPE_INT_t[:, ::1] _depth_mask_last
    cdef DTYPE_INT_t[:, ::1] _depth_mask_next

    # Dictionary of dimension names
    cdef object _dimension_names

    # Dictionary of variable names
    cdef object _variable_names

    # Dictionary containing tuples of variable shapes without time (e.g. {'u': (n_dpeth, n_latitude, n_longitude)})
    cdef object _variable_shapes

    # Dictionaries of variable dimension indices (e.g. {'u': {'depth': 0, 'latitude': 1, 'longitude': 2}})
    cdef object _variable_dimension_indices

    # Sea surface elevation
    cdef DTYPE_FLOAT_t[::1] _zeta_last
    cdef DTYPE_FLOAT_t[::1] _zeta_next
    
    # u/v/w velocity components
    cdef DTYPE_FLOAT_t[:,::1] _u_last
    cdef DTYPE_FLOAT_t[:,::1] _u_next
    cdef DTYPE_FLOAT_t[:,::1] _v_last
    cdef DTYPE_FLOAT_t[:,::1] _v_next
    cdef DTYPE_FLOAT_t[:,::1] _w_last
    cdef DTYPE_FLOAT_t[:,::1] _w_next
    
    # Vertical eddy diffusivities
    cdef DTYPE_FLOAT_t[:,::1] _Kz_last
    cdef DTYPE_FLOAT_t[:,::1] _Kz_next

    # Horizontal eddy viscosities
    cdef DTYPE_FLOAT_t[:,::1] _ah_last
    cdef DTYPE_FLOAT_t[:,::1] _ah_next

    # Wet/dry status of elements
    cdef DTYPE_INT_t[::1] _wet_cells_last
    cdef DTYPE_INT_t[::1] _wet_cells_next

    # Sea water potential temperature
    cdef DTYPE_FLOAT_t[:,::1] _thetao_last
    cdef DTYPE_FLOAT_t[:,::1] _thetao_next

    # Sea water salinity
    cdef DTYPE_FLOAT_t[:,::1] _so_last
    cdef DTYPE_FLOAT_t[:,::1] _so_next

    # Time direction
    cdef DTYPE_INT_t _time_direction

    # Time array
    cdef DTYPE_FLOAT_t _time_last
    cdef DTYPE_FLOAT_t _time_next

    # Options controlling the reading of eddy diffusivities
    cdef object _Kz_method_name, _Ah_method_name
    cdef DTYPE_INT_t _Kz_method, _Ah_method

    # Flag for surface only or full 3D transport
    cdef bint _surface_only

    # Flags that identify whether a given variable should be read in
    cdef bint _has_w, _has_is_wet,  _has_zeta

    # Smagorinsky parameters
    cdef DTYPE_FLOAT_t _C_smag

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
            self._dimension_names[dim_name] = self.config.get('OCEAN_DATA', config_name).strip()

        # Setup variable name mappings
        self._variable_names = {}
        var_config_names = {'uo': 'uo_var_name', 'vo': 'vo_var_name', 'wo': 'wo_var_name', 'zos': 'zos_var_name',
                            'Kz': 'Kz_var_name', 'Ah': 'Ah_var_name', 'thetao': 'thetao_var_name',
                            'so': 'so_var_name'}
        for var_name, config_name in var_config_names.items():
            try:
                var = self.config.get('OCEAN_DATA', config_name).strip()
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

        # Set options for handling the vertical eddy diffusivity
        self._Kz_method_name = self.config.get('OCEAN_DATA', 'Kz_method').strip().lower()
        if self._Kz_method_name not in ['none', 'file']:
            raise RuntimeError('Invalid option for `Kz_method` ({})'.format(self._Kz_method_name))

        if self._Kz_method_name == "none":
            self._Kz_method = 0
        elif self._Kz_method_name == "file":
            # Make sure a value for `Kz_var_name` has been provided.
            if 'Kz' not in self._variable_names:
                raise RuntimeError('The configuration file states that Kz data should be read from file \n'
                                    'yet a name for the Kz variable (`Kz_var_name`) has not been given \n'
                                    'or the config option was not included.')
            self._Kz_method = 1

        # Set options for handling the horizontal eddy diffusivity
        self._Ah_method_name = self.config.get('OCEAN_DATA', 'Ah_method').strip().lower()
        if self._Ah_method_name not in ['none', 'file', 'smagorinsky']:
            raise RuntimeError('Invalid option for `Ah_method` ({})'.format(self._Ah_method_name))

        if self._Ah_method_name == "none":
            self._Ah_method = 0
        elif self._Ah_method_name == "file":
            # Make sure a value for `Ah_var_name` has been provided.
            if 'Ah' not in self._variable_names:
                raise RuntimeError('The configuration file states that Ah data should be read from file \n'
                                    'yet a name for the Ah variable (`Ah_var_name`) has not been given \n'
                                    'or the config option was not included.')
            self._Ah_method = 1
        elif self._Ah_method_name == "smagorinsky":
            self._Ah_method = 2

            # Read in Smagorinsky parameters
            self._C_smag = self.config.getfloat('SMAGORINSKY', 'constant')

        # Has is wet flag?
        self._has_is_wet = self.config.getboolean("OCEAN_DATA", "has_is_wet")

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

    cdef get_boundary_intersection(self,
                                   Particle *particle_old,
                                   Particle *particle_new,
                                   DTYPE_FLOAT_t start_point[2],
                                   DTYPE_FLOAT_t end_point[2],
                                   DTYPE_FLOAT_t intersection[2]):
        """ Find the boundary intersection point

        This function is a wrapper for the same function implemented in UnstructuredGrid.

        Parameters
        ----------
        particle_old: *Particle
            The particle at its old position.

        particle_new: *Particle
            The particle at its new position.

        start_point : C array, float
            Start coordinates of the side the particle crossed.

        end_point : C array, float
            End coordinates of the side the particle crossed.

        intersection : C array, float
            Coordinates of the intersection point.

        Returns
        -------
        """
        return self._unstructured_grid.get_boundary_intersection(particle_old, particle_new, start_point, end_point,
                                                                 intersection)

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
            depth_upper_level = self._unstructured_grid.interpolate_in_time_and_space(self._depth_levels_last,
                                                                                      self._depth_levels_next,
                                                                                      k,
                                                                                      time_fraction,
                                                                                      particle)

            depth_lower_level = self._unstructured_grid.interpolate_in_time_and_space(self._depth_levels_last,
                                                                                      self._depth_levels_next,
                                                                                      k+1,
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
        depth_upper_level = self._unstructured_grid.interpolate_in_time_and_space(self._depth_levels_last,
                                                                                  self._depth_levels_next,
                                                                                  k,
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
        depth_upper_level = self._unstructured_grid.interpolate_in_time_and_space(self._depth_levels_last,
                                                                                  self._depth_levels_next,
                                                                                  k,
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

        if self._surface_only:
            return 0.0

        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        zeta = self._unstructured_grid.interpolate_in_time_and_space_2D(self._zeta_last,
                                                                        self._zeta_next,
                                                                        time_fraction,
                                                                        particle)

        return zeta

    cdef void get_velocity(self, DTYPE_FLOAT_t time, Particle* particle,
            DTYPE_FLOAT_t vel[3]) except +:
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

    cdef DTYPE_FLOAT_t get_horizontal_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            Particle* particle) except FLOAT_ERR:
        """ Returns the horizontal eddy diffusivity through linear interpolation

        Parameters
        ----------
        time : float
            Time at which to interpolate.

        particle: *Particle
            Pointer to a Particle object.

        Returns
        -------
        viscofh : float
            The interpolated value of the horizontal eddy diffusivity at the specified point in time and space.
        """
        cdef DTYPE_FLOAT_t var  # ah at (t, x1, x2, x3)

        if self._Ah_method == 1:
            var = self._get_variable(self._ah_last, self._ah_next, time, particle)
        elif self._Ah_method == 2:
            var = self._compute_smagorinsky_eddy_diffusivity(time, particle)
        else:
            raise RuntimeError('This dataset does not contain horizontal eddy viscosities.')

        return var

    cdef DTYPE_FLOAT_t _compute_smagorinsky_eddy_diffusivity(self, const DTYPE_FLOAT_t &time,
                                                             Particle* particle) except FLOAT_ERR:
        """ Compute horizontal eddy diffusivity term from Smagorinsky expression

        The approach here is to compute the eddy diffusivity from the velocity field
        using the Smagorinsky. The diffusivity is assumed to be isotropic is x and y.

        Parameters
        ----------
        time : float
            Time at which to interpolate.

        particle: *Particle
            Pointer to a Particle object.

        Returns
        -------
         : float
             The horizontal eddy diffusivity coefficient.
        """

            # Interpolated values on lower and upper bounding depth levels
        cdef DTYPE_FLOAT_t Kz_level_1
        cdef DTYPE_FLOAT_t Kz_level_2
        cdef DTYPE_FLOAT_t Kz

        # Particle k_layer
        cdef DTYPE_INT_t k_layer = particle.get_k_layer()

        cdef DTYPE_FLOAT_t time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        # No vertical interpolation for particles near to the bottom
        if particle.get_in_vertical_boundary_layer() is True:
            Kz = self._compute_smagorinsky_eddy_diffusivity_on_level(time_fraction, k_layer, particle)

        else:
            Kz_level_1 = self._compute_smagorinsky_eddy_diffusivity_on_level(time_fraction, k_layer+1, particle)
            Kz_level_2 = self._compute_smagorinsky_eddy_diffusivity_on_level(time_fraction, k_layer, particle)

            Kz = interp.linear_interp(particle.get_omega_interfaces(), Kz_level_1, Kz_level_2)

        return Kz

    cdef DTYPE_FLOAT_t _compute_smagorinsky_eddy_diffusivity_on_level(self, const DTYPE_FLOAT_t &time_fraction,
                                                                      const DTYPE_INT_t k,
                                                                      Particle* particle) except FLOAT_ERR:
        """ Compute horizontal eddy diffusivity using Smagorinsky expression on level k

        Kz = A_e * (1/Pr) * C * sqrt((du/dx)^2 + (dv/dy)^2 + 0.5*(du/dy + dv/dx)^2)

        where A_e is the area of the element within which the particle resides, Pr is the Prandtl number and
        C is a constant parameter. The Prandtl number is taken as 1.0.
        """
        # Gradients in u and v ([du/dx, du/dy] and [dv/dx, dv/dy])
        cdef DTYPE_FLOAT_t u_prime[2]
        cdef DTYPE_FLOAT_t v_prime[2]

        # Element area
        cdef DTYPE_FLOAT_t A_e

        # Intermediate terms
        cdef DTYPE_FLOAT_t term_1, term_2

        # Get the element's area
        A_e = self._unstructured_grid.get_element_area(particle)

        # Compute velocity gradient terms
        self._unstructured_grid.interpolate_grad_in_time_and_space(self._u_last, self._u_next, k, time_fraction,
                                                                   particle, u_prime)
        self._unstructured_grid.interpolate_grad_in_time_and_space(self._v_last, self._v_next, k, time_fraction,
                                                                   particle, v_prime)

        # Compute Smagorinsky term and return
        term_1 = u_prime[0]**2 + v_prime[1]**2
        term_2 = 0.5 * (u_prime[1] + v_prime[0])**2

        return 0.5 * self._C_smag * A_e * sqrt(term_1 + term_2)

    cdef void get_horizontal_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t time,
            Particle* particle, DTYPE_FLOAT_t Ah_prime[2]) except +:
        """ Returns the gradient in the horizontal eddy diffusivity

        Parameters
        ----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.

        Ah_prime : C array, float
            dAh_dx and dAh_dy components stored in a C array of length two.

        References
        ----------
        Lynch, D. R. et al (2014). Particles in the coastal ocean: theory and
        applications. Cambridge: Cambridge University Press.
        doi.org/10.1017/CBO9781107449336
        """
        # Variables used in interpolation in time
        cdef DTYPE_FLOAT_t time_fraction

        # Particle k_layer
        cdef DTYPE_INT_t k_layer

        # Gradients in Ah on lower and upper bounding levels
        cdef DTYPE_FLOAT_t Ah_prime_level_1[2]
        cdef DTYPE_FLOAT_t Ah_prime_level_2[2]

        if self._Ah_method == 1:
            # Time fraction
            time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

            # K layer
            k_layer = particle.get_k_layer()

            # No vertical interpolation for particles near to the bottom,
            if particle.get_in_vertical_boundary_layer() is True:
                self._unstructured_grid.interpolate_grad_in_time_and_space(self._ah_last, self._ah_next, k_layer,
                                                                           time_fraction, particle, Ah_prime)
            else:
                self._unstructured_grid.interpolate_grad_in_time_and_space(self._ah_last, self._ah_next, k_layer+1,
                                                                           time_fraction, particle, Ah_prime_level_1)
                self._unstructured_grid.interpolate_grad_in_time_and_space(self._ah_last, self._ah_next, k_layer,
                                                                           time_fraction, particle, Ah_prime_level_2)
                Ah_prime[0] = interp.linear_interp(particle.get_omega_interfaces(), Ah_prime_level_1[0], Ah_prime_level_2[0])
                Ah_prime[1] = interp.linear_interp(particle.get_omega_interfaces(), Ah_prime_level_1[1], Ah_prime_level_2[1])

        elif self._Ah_method == 2:
            # With C0 continuity, Ah is constant within an element when computed from velocity component derivatives.
            # To improve this, will need to compute derivatives in a different way (e.g. using a least squares method).
            # Note this is really an issue for UnstructuredGrid, which is responsible for computing derivatives.
            Ah_prime[0] = 0.0
            Ah_prime[1] = 0.0

        else:
            raise RuntimeError('This dataset does not contain horizontal eddy viscosities.')

        return

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
        Kz : float
            The vertical eddy diffusivity.        
        
        """
        cdef DTYPE_FLOAT_t var  # Kz at (t, x1, x2, x3)

        if self._Kz_method == 1:
            if not self._surface_only:
                var = self._get_variable(self._Kz_last, self._Kz_next, time, particle)
            else:
                var = 0.0
        else:
            raise RuntimeError('This dataset does not contain vertical eddy diffusivities.')

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
        compute depths and Kz at all levels, then interpolate between then. Would
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
        cdef DTYPE_FLOAT_t Kz_0, Kz_1, Kz_2, Kz_3
        cdef DTYPE_FLOAT_t z_0, z_1, z_2, z_3
        cdef DTYPE_FLOAT_t dKz_lower_level, dKz_upper_level

        # Particle k_layer
        cdef DTYPE_INT_t k_layer = particle.get_k_layer()

        # The value of the derivative
        cdef DTYPE_FLOAT_t dKz_dz

        if self._Kz_method == 1:
            if self._surface_only:
                dKz_dz = 0.0
            else:
                raise NotImplementedError('Implementation is under development')
        else:
            raise RuntimeError('This dataset does not contain vertical eddy diffusivities.')

        return dKz_dz

#        if particle.get_in_vertical_boundary_layer() == True:
#            Kz_0 = self._get_variable_on_level(self._Kz_last, self._Kz_next, time, particle, k_layer-1)
#            z_0 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer-1)
#
#            Kz_1 = self._get_variable_on_level(self._Kz_last, self._Kz_next, time, particle, k_layer)
#            z_1 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer)
#
#            dKz_upper_level = (Kz_0 - Kz_1) / (z_0 - z_1)
#
#            return dKz_upper_level
#
#        if k_layer == 0:
#            Kz_0 = self._get_variable_on_level(self._Kz_last, self._Kz_next, time, particle, k_layer)
#            z_0 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer)
#
#            Kz_1 = self._get_variable_on_level(self._Kz_last, self._Kz_next, time, particle, k_layer+1)
#            z_1 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer+1)
#
#            Kz_2 = self._get_variable_on_level(self._Kz_last, self._Kz_next, time, particle, k_layer+2)
#            z_2 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer+2)
#
#            dKz_lower_level = (Kz_0 - Kz_2) / (z_0 - z_2)
#            dKz_upper_level = (Kz_0 - Kz_1) / (z_0 - z_1)
#
#        elif k_layer == self._n_depth - 2:
#            Kz_0 = self._get_variable_on_level(self._Kz_last, self._Kz_next, time, particle, k_layer-1)
#            z_0 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer-1)
#
#            Kz_1 = self._get_variable_on_level(self._Kz_last, self._Kz_next, time, particle, k_layer)
#            z_1 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer)
#
#            Kz_2 = self._get_variable_on_level(self._Kz_last, self._Kz_next, time, particle, k_layer+1)
#            z_2 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer+1)
#
#            dKz_lower_level = (Kz_1 - Kz_2) / (z_1 - z_2)
#            dKz_upper_level = (Kz_0 - Kz_2) / (z_0 - z_2)
#
#        else:
#            Kz_0 = self._get_variable_on_level(self._Kz_last, self._Kz_next, time, particle, k_layer-1)
#            z_0 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer-1)
#
#            Kz_1 = self._get_variable_on_level(self._Kz_last, self._Kz_next, time, particle, k_layer)
#            z_1 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer)
#
#            Kz_2 = self._get_variable_on_level(self._Kz_last, self._Kz_next, time, particle, k_layer+1)
#            z_2 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer+1)
#
#            Kz_3 = self._get_variable_on_level(self._Kz_last, self._Kz_next, time, particle, k_layer+2)
#            z_3 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k_layer+2)
#
#            dKz_lower_level = (Kz_1 - Kz_3) / (z_1 - z_3)
#            dKz_upper_level = (Kz_0 - Kz_2) / (z_0 - z_2)
#
#        return interp.linear_interp(particle.get_omega_interfaces(), dKz_lower_level, dKz_upper_level)

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

    cdef DTYPE_FLOAT_t _get_variable(self, DTYPE_FLOAT_t[:, ::1] var_last,
                                     DTYPE_FLOAT_t[:, ::1] var_next,
                                     const DTYPE_FLOAT_t &time, Particle* particle) except FLOAT_ERR:
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
            return self._unstructured_grid.interpolate_in_time_and_space(var_last,
                                                                         var_next,
                                                                         k_layer,
                                                                         time_fraction,
                                                                         particle)
        else:
            var_level_1 = self._unstructured_grid.interpolate_in_time_and_space(var_last,
                                                                                var_next,
                                                                                k_layer+1,
                                                                                time_fraction,
                                                                                particle)

            var_level_2 = self._unstructured_grid.interpolate_in_time_and_space(var_last,
                                                                                var_next,
                                                                                k_layer,
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
        coordinate_system = self.config.get("SIMULATION",
                                            "coordinate_system").strip().lower()

        if coordinate_system == "geographic":
            x = self.mediator.get_grid_variable('longitude', (self._n_nodes), DTYPE_FLOAT)
            y = self.mediator.get_grid_variable('latitude', (self._n_nodes), DTYPE_FLOAT)
            xc = self.mediator.get_grid_variable('longitude_c', (self._n_elems), DTYPE_FLOAT)
            yc = self.mediator.get_grid_variable('latitude_c', (self._n_elems), DTYPE_FLOAT)

            # Convert to radians
            x = x * deg_to_radians
            y = y * deg_to_radians
            xc = xc * deg_to_radians
            yc = yc * deg_to_radians

            # Don't apply offsets in geographic case - set them to 0.0!
            self._xmin = 0.0
            self._ymin = 0.0
        else:
            raise PyLagValueError(f"Unsupported model coordinate system "
                             f"`{coordinate_system}'")

        # Land sea mask
        self._land_sea_mask_c = self.mediator.get_grid_variable('mask_c', (self._n_elems), DTYPE_INT)
        self._land_sea_mask = self.mediator.get_grid_variable('mask', (self._n_nodes), DTYPE_INT)

        # Element areas
        areas = self.mediator.get_grid_variable('area', (self._n_elems), DTYPE_FLOAT)

        # Initialise unstructured grid
        self._unstructured_grid = get_unstructured_grid(self.config, self._name, self._n_nodes, self._n_elems,
                                                        self._nv, self._nbe, x, y, xc, yc, self._land_sea_mask_c,
                                                        self._land_sea_mask, areas=areas)

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
        
        # Initialise zeta with zero values
        self._zeta_last = np.zeros(self._n_nodes, dtype=DTYPE_FLOAT)
        self._zeta_next = np.zeros(self._n_nodes, dtype=DTYPE_FLOAT)

        # If zeta is being read in from file, add zeta shape and
        # dimension indices to the appropriate dictionaries
        if self._has_zeta:
            self._variable_shapes['zos'] = self.mediator.get_variable_shape(
                self._variable_names['zos'],
                include_time=False)
            dimensions = self.mediator.get_variable_dimensions(
                self._variable_names['zos'],
                include_time=False)
            self._variable_dimension_indices['zos'] = {
                'latitude': dimensions.index(self._dimension_names['latitude']),
                'longitude': dimensions.index(self._dimension_names['longitude'])}

        # Add other vars to shape and dimension indices dictionaries
        var_names = ['uo', 'vo']
        if self._has_w: var_names.append('wo')
        if self._Kz_method == 1: var_names.append('Kz')
        if self._Ah_method == 1: var_names.append('Ah')
        if 'thetao' in self.env_var_names: var_names.append('thetao')
        if 'so' in self.env_var_names: var_names.append('so')
        for var_name in var_names:
            self._variable_shapes[var_name] = self.mediator.get_variable_shape(
                self._variable_names[var_name],
                include_time=False)
            dimensions = self.mediator.get_variable_dimensions(
                self._variable_names[var_name],
                include_time=False)

            # Allow for the possibility that the depth dimension is not present
            if self._dimension_names['depth'] in dimensions:
                self._variable_dimension_indices[var_name] = {
                    'depth': dimensions.index(self._dimension_names['depth']),
                    'latitude': dimensions.index(self._dimension_names['latitude']),
                    'longitude': dimensions.index(self._dimension_names['longitude'])}
            else:
                self._variable_dimension_indices[var_name] = {
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
        self._u_last = self._reshape_var(u_last, self._variable_dimension_indices['uo'],
            add_depth_axis=True)
        del(u_last)

        u_next = self.mediator.get_time_dependent_variable_at_next_time_index(u_var_name,
                self._variable_shapes['uo'], DTYPE_FLOAT)
        self._u_next = self._reshape_var(u_next, self._variable_dimension_indices['uo'],
            add_depth_axis=True)
        del(u_next)

        # Update memory views for v
        v_var_name = self._variable_names['vo']
        v_last = self.mediator.get_time_dependent_variable_at_last_time_index(v_var_name,
                self._variable_shapes['vo'], DTYPE_FLOAT)
        self._v_last = self._reshape_var(v_last, self._variable_dimension_indices['vo'],
            add_depth_axis=True)
        del(v_last)

        v_next = self.mediator.get_time_dependent_variable_at_next_time_index(v_var_name,
                self._variable_shapes['vo'], DTYPE_FLOAT)
        self._v_next = self._reshape_var(v_next, self._variable_dimension_indices['vo'],
            add_depth_axis=True)
        del(v_next)

        # Update memory views for w
        if self._has_w:
            w_var_name = self._variable_names['wo']
            w_last = self.mediator.get_time_dependent_variable_at_last_time_index(w_var_name,
                    self._variable_shapes['wo'], DTYPE_FLOAT)
            self._w_last = self._reshape_var(w_last, self._variable_dimension_indices['wo'],
                add_depth_axis=True)
            del(w_last)

            w_next = self.mediator.get_time_dependent_variable_at_next_time_index(w_var_name,
                    self._variable_shapes['wo'], DTYPE_FLOAT)
            self._w_next = self._reshape_var(w_next, self._variable_dimension_indices['wo'],
                add_depth_axis=True)
            del(w_next)

        # Update depth mask
        if not self._surface_only:
            depth_mask_last = self.mediator.get_mask_at_last_time_index(u_var_name,
                    self._variable_shapes['uo'])
            self._depth_mask_last = self._reshape_var(depth_mask_last, self._variable_dimension_indices['uo'])
            del(depth_mask_last)

            depth_mask_next = self.mediator.get_mask_at_next_time_index(u_var_name,
                    self._variable_shapes['uo'])
            self._depth_mask_next = self._reshape_var(depth_mask_next, self._variable_dimension_indices['uo'])
            del(depth_mask_next)

            # Compute actual depth levels using reference values and zeta
            for k in xrange(self._n_depth):
                for i in xrange(self._n_nodes):
                    self._depth_levels_last[k, i] = self._reference_depth_levels[k] + self._zeta_last[i]
                    self._depth_levels_next[k, i] = self._reference_depth_levels[k] + self._zeta_next[i]

        # Update memory views for Kz
        if self._Kz_method == 1:
            Kz_var_name = self._variable_names['Kz']
            Kz_last = self.mediator.get_time_dependent_variable_at_last_time_index(Kz_var_name,
                    self._variable_shapes['Kz'], DTYPE_FLOAT)
            self._Kz_last = self._reshape_var(Kz_last, self._variable_dimension_indices['Kz'],
                add_depth_axis=True)
            del(Kz_last)

            Kz_next = self.mediator.get_time_dependent_variable_at_next_time_index(Kz_var_name,
                    self._variable_shapes['Kz'], DTYPE_FLOAT)
            self._Kz_next = self._reshape_var(Kz_next, self._variable_dimension_indices['Kz'],
                add_depth_axis=True)
            del(Kz_next)

        # Update memory views for Ah
        if self._Ah_method == 1:
            ah_var_name = self._variable_names['Ah']
            ah_last = self.mediator.get_time_dependent_variable_at_last_time_index(ah_var_name,
                    self._variable_shapes['Ah'], DTYPE_FLOAT)
            self._ah_last = self._reshape_var(ah_last, self._variable_dimension_indices['Ah'],
                add_depth_axis=True)
            del(ah_last)

            ah_next = self.mediator.get_time_dependent_variable_at_next_time_index(ah_var_name,
                    self._variable_shapes['Ah'], DTYPE_FLOAT)
            self._ah_next = self._reshape_var(ah_next, self._variable_dimension_indices['Ah'],
                add_depth_axis=True)
            del(ah_next)

        # Set is wet status
        # NB the status of cells is inferred from the depth mask and the land-sea element mask. If a surface cell is
        # masked but it is not a land cell, then it is assumed to be dry.
        if not self._surface_only:
            for i in xrange(self._n_elems):
                if self._land_sea_mask_c[i] == 0:
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

            thetao_last = self.mediator.get_time_dependent_variable_at_last_time_index(var_name,
                    self._variable_shapes['thetao'], DTYPE_FLOAT)
            self._thetao_last = self._reshape_var(thetao_last, self._variable_dimension_indices['thetao'],
                    add_depth_axis=True)
            del(thetao_last)
            
            thetao_next = self.mediator.get_time_dependent_variable_at_next_time_index(var_name,
                    self._variable_shapes['thetao'], DTYPE_FLOAT)
            self._thetao_next = self._reshape_var(thetao_next, self._variable_dimension_indices['thetao'],
                    add_depth_axis=True)
            del(thetao_next)

        if 'so' in self.env_var_names:
            var_name = self._variable_names['so']

            so_last = self.mediator.get_time_dependent_variable_at_last_time_index(var_name,
                    self._variable_shapes['so'], DTYPE_FLOAT)
            self._so_last = self._reshape_var(so_last, self._variable_dimension_indices['so'],
                    add_depth_axis=True)
            del(so_last)

            so_next = self.mediator.get_time_dependent_variable_at_next_time_index(var_name,
                    self._variable_shapes['so'], DTYPE_FLOAT)
            self._so_next = self._reshape_var(so_next, self._variable_dimension_indices['so'],
                    add_depth_axis=True)
            del(so_next)

        return

    def _reshape_var(self, var, dimension_indices, add_depth_axis=False):
        """ Reshape variable for PyLag

        Variables with the following dimensions are supported:

        2D - [lat, lon] in any order

        3D - [depth, lat, lon] in any order

        Latitude trimming is applied as required. If the variable is 2D
        and add_depth_axis is True, then a new axis is added at the start
        of the array.

        Parameters
        ----------
        var : NDArray
            The variable to sort

        dimension_indices : dict
            Dictionary of dimension indices

        add_depth_axis : bool
            If True, add a new axis at the start of the array

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

            # Add depth axis if required
            if add_depth_axis:
                var = var[np.newaxis, :, :]

            return np.ascontiguousarray(var.reshape(np.prod(var.shape), order='C')[self._permutation])

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

            return np.ascontiguousarray(var.reshape(var.shape[0], np.prod(var.shape[1:]), order='C')[:, self._permutation])
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

