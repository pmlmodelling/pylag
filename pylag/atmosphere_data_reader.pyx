"""
Atmosphere data reader for reading atmospheric data on an Arakawa A-grid

Note
----
atmosphere_data_reader is implemented in Cython. Only a small portion of the
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
from pylag.unstructured import get_unstructured_grid
from pylag.numerics import get_time_direction
from pylag.exceptions import PyLagValueError


cdef class AtmosphereDataReader(DataReader):
    """ AtmosphereDataReader for atmospheric inputs on a Arakawa A-grid

    Objects of type AtmosphereDataReader are intended to manage all access to
    atmospheric data defined on a Arakawa A-grid, including data describing the
    grid as well as atmospheric variables. Provided are methods for searching
    the model grid for host horizontal elements and for interpolating gridded
    field data to a given point in space and time.

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

    # Dictionary of dimension names
    cdef object _dimension_names

    # Dictionary of variable names
    cdef object _variable_names

    # Dictionary containing tuples of variable shapes without time
    # (e.g. {'u': (n_dpeth, n_latitude, n_longitude)})
    cdef object _variable_shapes

    # Dictionaries of variable dimension indices
    # (e.g. {'u': {'latitude': 0, 'longitude': 1}})
    cdef object _variable_dimension_indices

    # Land sea mask on elements (1 - sea point, 0 - land point)
    cdef DTYPE_INT_t[::1] _land_sea_mask_c
    cdef DTYPE_INT_t[::1] _land_sea_mask

    # u/v/w velocity components
    cdef DTYPE_FLOAT_t[::1] _u10_last
    cdef DTYPE_FLOAT_t[::1] _u10_next
    cdef DTYPE_FLOAT_t[::1] _v10_last
    cdef DTYPE_FLOAT_t[::1] _v10_next

    # Time direction
    cdef DTYPE_INT_t _time_direction

    # Time array
    cdef DTYPE_FLOAT_t _time_last
    cdef DTYPE_FLOAT_t _time_next

    def __init__(self, config, mediator):
        self.config = config
        self.mediator = mediator

        self._name = b'atmosphere'

        # Time direction
        self._time_direction = <int>get_time_direction(config)

        # Setup dimension name mappings
        self._dimension_names = {}
        dim_config_names = {'time': 'time_dim_name',
                'latitude': 'latitude_dim_name',
                'longitude': 'longitude_dim_name'}
        for dim_name, config_name in dim_config_names.items():
            self._dimension_names[dim_name] = \
                    self.config.get('ATMOSPHERE_DATA',
                    config_name).strip()

        # Setup variable name mappings
        self._variable_names = {}
        var_config_names = {'u10': 'u10_var_name', 'v10': 'v10_var_name'}
        for var_name, config_name in var_config_names.items():
            try:
                var = self.config.get('ATMOSPHERE_DATA',
                        config_name).strip()
                if var:
                    self._variable_names[var_name] = var
            except (configparser.NoOptionError) as e:
                pass

        # Initialise dictionaries for variable shapes and dimension indices
        self._variable_shapes = {}
        self._variable_dimension_indices = {}

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

        time_fraction = interp.get_linear_fraction(time, self._time_last,
                self._time_next)
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

        # Use the old host to start a local search
        host = particle_new.get_host_horizontal_elem(self._name)
        particle_new.set_host_horizontal_elem(self._name,
                particle_old.get_host_horizontal_elem(self._name))
        flag = self._unstructured_grid.find_host_using_local_search(
                particle_new)

        if flag == IN_DOMAIN:
            return flag

        # Local search failed to find the particle. Perform check to see if
        # the particle has indeed left the model domain. Reset the host first
        # in order to preserve the original state of particle_new.
        particle_new.set_host_horizontal_elem(self._name, host)
        flag = self._unstructured_grid.find_host_using_particle_tracing(
                particle_old, particle_new)

        return flag

    cdef DTYPE_INT_t find_host_using_local_search(self,
            Particle *particle) except INT_ERR:
        """ Returns the host horizontal element through local searching.

        This function is a wrapper for the same function implemented in
        UnstructuredGrid.

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

    cdef DTYPE_INT_t find_host_using_global_search(self,
            Particle *particle) except INT_ERR:
        """ Returns the host horizontal element through global searching.

        This function is a wrapper for the same function implemented in
        UnstructuredGrid.

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

    cdef void get_ten_meter_wind_velocity(self, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t wind_vel[2]) except +:
        """ Returns the ten meter wind velocity

        Parameters:
        -----------
        time : float
            Time at which to interpolate.

        particle: *Particle
            Pointer to a Particle object.

        wind_vel : C array, float
            Horizontal wind velocity components in C array of length two.
        """
        wind_vel[0] = self._get_variable(self._u10_last, self._u10_next, time,
                particle)
        wind_vel[1] = self._get_variable(self._v10_last, self._v10_next, time,
                particle)

        return

    cdef DTYPE_FLOAT_t _get_variable(self, DTYPE_FLOAT_t[::1] var_last,
            DTYPE_FLOAT_t[::1] var_next, const DTYPE_FLOAT_t &time,
            Particle* particle) except FLOAT_ERR:
        """ Returns the value of the variable through linear interpolation

        Private method for interpolating fields specified at element
        nodes on depth levels. For particle at depths above h and
        above a lower level with masked nodes, extrapolation is used.

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
            The interpolated value of the variable at the specified
            point in time and space.
        """
        cdef DTYPE_FLOAT_t time_fraction
        cdef DTYPE_FLOAT_t var

        time_fraction = interp.get_linear_fraction_safe(time,
                self._time_last, self._time_next)

        var = self._unstructured_grid.interpolate_in_time_and_space_2D(
                var_last, var_next, time_fraction, particle)

        return var

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

        # Read in flags signifying whether time dependent variable arrays
        # should be trimmed in order to eliminate gridded data at -90 deg.
        # or 90 deg. N.
        self._trim_first_latitude = self.mediator.get_grid_variable(
                'trim_first_latitude', (1), DTYPE_INT)
        self._trim_last_latitude = self.mediator.get_grid_variable(
                'trim_last_latitude', (1), DTYPE_INT)

        # Grid connectivity/adjacency
        self._nv = self.mediator.get_grid_variable('nv',
                (3, self._n_elems), DTYPE_INT)
        self._nbe = self.mediator.get_grid_variable('nbe',
                (3, self._n_elems), DTYPE_INT)

        # Node permutation
        self._permutation = self.mediator.get_grid_variable('permutation',
                (self._n_nodes), DTYPE_INT)

        x = self.mediator.get_grid_variable('longitude', (self._n_nodes),
                DTYPE_FLOAT)
        y = self.mediator.get_grid_variable('latitude', (self._n_nodes),
                DTYPE_FLOAT)
        xc = self.mediator.get_grid_variable('longitude_c', (self._n_elems),
                DTYPE_FLOAT)
        yc = self.mediator.get_grid_variable('latitude_c', (self._n_elems),
                DTYPE_FLOAT)

        # Convert to radians
        x = x * deg_to_radians
        y = y * deg_to_radians
        xc = xc * deg_to_radians
        yc = yc * deg_to_radians

        # Land sea mask
        try:
            self._land_sea_mask_c = self.mediator.get_grid_variable('mask_c',
                    (self._n_elems), DTYPE_INT)
        except KeyError:
            # No mask - treat all points as being sea.
            self._land_sea_mask_c = np.zeros(self._n_elems, dtype=DTYPE_INT)

        try:
            self._land_sea_mask = self.mediator.get_grid_variable('mask',
                    (self._n_nodes), DTYPE_INT)
        except KeyError:
            # No mask - treat all points as being sea.
            self._land_sea_mask = np.zeros(self._n_nodes, dtype=DTYPE_INT)

        # Element areas
        areas = self.mediator.get_grid_variable('area', (self._n_elems),
                DTYPE_FLOAT)

        # Initialise unstructured grid
        self._unstructured_grid = get_unstructured_grid(self.config,
                self._name, self._n_nodes, self._n_elems, self._nv, self._nbe,
                x, y, xc, yc, self._land_sea_mask_c, self._land_sea_mask,
                areas=areas)

        # Add 3D vars to shape and dimension indices dictionaries
        var_names = ['uo', 'vo']
        for var_name in var_names:
            self._variable_shapes[var_name] = \
                    self.mediator.get_variable_shape(self._variable_names[var_name])[1:]
            dimensions = \
                    self.mediator.get_variable_dimensions(self._variable_names[var_name])[1:]
            self._variable_dimension_indices[var_name] = \
                    {'depth': dimensions.index(self._dimension_names['depth']),
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

        # Update time references
        self._time_last = self.mediator.get_time_at_last_time_index()
        self._time_next = self.mediator.get_time_at_next_time_index()

        # Update memory views for u10
        u10_var_name = self._variable_names['u10']
        u10_last = self.mediator.get_time_dependent_variable_at_last_time_index(
                u10_var_name, self._variable_shapes['u10'], DTYPE_FLOAT)
        self._u10_last = self._reshape_var(u10_last,
                self._variable_dimension_indices['u10'])
        del(u10_last)

        u10_next = self.mediator.get_time_dependent_variable_at_next_time_index(
                u10_var_name, self._variable_shapes['u10'], DTYPE_FLOAT)
        self._u10_next = self._reshape_var(u10_next,
                self._variable_dimension_indices['u10'])
        del(u10_next)

        # Update memory views for v10
        v10_var_name = self._variable_names['v10']
        v10_last = self.mediator.get_time_dependent_variable_at_last_time_index(
                v10_var_name, self._variable_shapes['v10'], DTYPE_FLOAT)
        self._v10_last = self._reshape_var(v10_last,
                self._variable_dimension_indices['v10'])
        del(v10_last)

        v10_next = self.mediator.get_time_dependent_variable_at_next_time_index(
                v10_var_name, self._variable_shapes['v10'], DTYPE_FLOAT)
        self._v10_next = self._reshape_var(v10_next,
                self._variable_dimension_indices['v10'])
        del(v10_next)

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

            return np.ascontiguousarray(var.reshape(var.shape[0],
                    np.prod(var.shape[1:]), order='C')[:, self._permutation])
        else:
            raise PyLagValueError(f'Unsupported number of dimensions '
                                  f'{n_dimensions}')