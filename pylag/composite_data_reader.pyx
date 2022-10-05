"""
Data reader for multiple sources of input data (ocean, wind, waves).

Note
----
composite_data_reader is implemented in Cython. Only a small portion of the
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

from collections import Counter
import numpy as np

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cython imports
from pylag.data_reader cimport DataReader
from pylag.particle cimport Particle

# PyLag python imports
from pylag.exceptions import PyLagRuntimeError


cdef class CompositeDataReader(DataReader):
    """ DataReader for multiple sources of input data

    Objects of type `CompositeDataReader` can be used to read in data from
    from multiple sources. The class structure follows the canonical
    Composite design pattern, in which both `CompositeDataReader` and
    the separate concrete data readers (class variables) it manages inherit
    a common interface from an abstract base class (`DataReader`).

    CompositeDataReaders can be used to read in ocean data with atmospheric
    and wave data, where the latter may be stored on a different grid.
    CompositeDataReaders coordinate the reading in of all data, and direct
    data access requests to the appropriate data reader.

    Parameters
    ----------
    config : ConfigParser
        Configuration object.
    
    ocean_data_reader : DataReader
        Ocean data reader. Optional, default None.

    atmosphere_data_reader : DataReader
        Atmosphere data reader. Optional, default None.

    waves_data_reader : DataReader
        Waves data reader. Optional, default None.
    """
    
    # Configuration object
    cdef object config

    # Data readers
    cdef DataReader ocean_data_reader
    cdef DataReader atmosphere_data_reader
    cdef DataReader waves_data_reader

    cdef bint uising_atmosphere_data
    cdef bint using_waves_data

    def __init__(self, config, ocean_data_reader, atmosphere_data_reader=None,
            waves_data_reader=None):

        self.config = config

        # Check that a valid data reader has been provided
        if ocean_data_reader is None:
            raise PyLagRuntimeError(f'A valid ocean data reader must be '
                                    f'provided.')
        self.ocean_data_reader = ocean_data_reader

        if atmosphere_data_reader is not None:
            self.using_atmosphere_data = True
            self.atmosphere_data_reader = atmosphere_data_reader
        else:
            self.using_atmosphere_data = False
            self.atmosphere_data_reader = DataReader()

        if waves_data_reader is not None:
            self.using_waves_data = True
            self.waves_data_reader = waves_data_reader
        else:
            self.using_waves_data = False
            self.waves_data_reader = DataReader()

    cpdef get_grid_names(self):
        """ Return a list of grid names
        """
        all_grid_names = self.ocean_data_reader.get_grid_names()

        if self.using_atmosphere_data:
            all_grid_names += self.atmosphere_data_reader.get_grid_names()

        if self.using_waves_data:
            all_grid_names += self.waves_data_reader.get_grid_names()

        # Check for duplicate names
        most_common_name, count = Counter(all_grid_names).most_common(n=1)[0]
        if count > 1:
            raise PyLagRuntimeError(f'Twe (or more) input grids have the same '
                                    f'name. The most common name is '
                                    f'{most_common_name} which appears '
                                    f'{count} times.')

        return all_grid_names

    cpdef setup_data_access(self, start_datetime, end_datetime):
        """ Set up access to time-dependent variables.
        
        Parameters
        ----------
        start_datetime : Datetime
            Datetime object corresponding to the simulation start time.
        
        end_datetime : Datetime
            Datetime object corresponding to the simulation end time.
        """
        self.ocean_data_reader.setup_data_access(start_datetime,
                                                 end_datetime)

        if self.using_atmosphere_data:
            self.atmosphere_data_reader.setup_data_access(start_datetime,
                                                          end_datetime)

        if self.using_waves_data:
            self.waves_data_reader.setup_data_access(start_datetime,
                                                     end_datetime)

    cpdef read_data(self, DTYPE_FLOAT_t time):
        """ Read in time dependent variable data from file?

        `time` is used to test if new data should be read in from file. If this
        is the case, arrays containing time-dependent variable data are updated.

        Parameters
        ----------
        time : float
            The current time.
        """
        self.ocean_data_reader.read_data(time)

        if self.using_atmosphere_data:
            self.atmosphere_data_reader.read_data(time)

        if self.using_waves_data:
            self.waves_data_reader.read_data(time)

    cdef DTYPE_INT_t find_host(self, Particle *particle_old,
                               Particle *particle_new) except INT_ERR:
        """ Set the host horizontal element on each input grid

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
            in is masked.

        Parameters:
        -----------
        particle_old: *Particle
            The particle at its old position.

        particle_new: *Particle
            The particle at its new position. The host element will be updated.

        Returns
        -------
        flag : int
            Integer flag that indicates whether or not the seach was successful.
        """
        cdef DTYPE_INT_t flag

        flag = self.ocean_data_reader.find_host(particle_old,
                                                particle_new)

        if flag != IN_DOMAIN:
            return flag

        if self.using_atmosphere_data:
            flag = self.atmosphere_data_reader.find_host(particle_old,
                                                         particle_new)

            if flag != IN_DOMAIN:
                return flag

        if self.using_waves_data:
            flag = self.waves_data_reader.find_host(particle_old,
                                                    particle_new)

            if flag != IN_DOMAIN:
                return flag

        return flag

    cdef DTYPE_INT_t find_host_using_local_search(self, Particle *particle) except INT_ERR:
        """ Sets host horizontal using local searching.

        Parameters
        ----------
        particle: *Particle
            The particle.

        Returns:
        --------
        flag : int
            Integer flag that indicates whether or not the search was successful.
        """
        cdef DTYPE_INT_t flag

        flag = self.ocean_data_reader.find_host_using_local_search(particle)

        if flag != IN_DOMAIN:
            return flag

        if self.using_atmosphere_data:
            flag = self.atmosphere_data_reader.find_host_using_local_search(
                    particle)

            if flag != IN_DOMAIN:
                return flag

        if self.using_waves_data:
            flag = self.waves_data_reader.find_host_using_local_search(particle)

            if flag != IN_DOMAIN:
                return flag

        return flag

    cdef DTYPE_INT_t find_host_using_global_search(self, Particle *particle) except INT_ERR:
        """ Sets host horizontal elements through global searching

        Parameters
        ----------
        particle_old: *Particle
            The particle.

        Returns
        -------
        flag : int
            Integer flag that indicates whether or not the search was successful.
        """
        cdef DTYPE_INT_t flag = IN_DOMAIN

        flag = self.ocean_data_reader.find_host_using_global_search(particle)

        if flag != IN_DOMAIN:
            return flag

        if self.using_atmosphere_data:
            flag = self.atmosphere_data_reader.find_host_using_global_search(particle)

            if flag != IN_DOMAIN:
                return flag

        if self.using_waves_data:
            flag = self.waves_data_reader.find_host_using_global_search(particle)

            if flag != IN_DOMAIN:
                return flag

        return flag

    cdef get_boundary_intersection(self,
                                   Particle *particle_old,
                                   Particle *particle_new,
                                   DTYPE_FLOAT_t start_point[2],
                                   DTYPE_FLOAT_t end_point[2],
                                   DTYPE_FLOAT_t intersection[2]):
        """ Find the boundary intersection point

        Boundary intersection points mark the location where the particle
        crossed over from the water to the land. As such, boundary
        intersection points are only computed on the ocean grid.

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
        self.ocean_data_reader.get_boundary_intersection(particle_old,
                                                         particle_new,
                                                         start_point,
                                                         end_point,
                                                         intersection)

    cdef set_default_location(self, Particle *particle):
        """ Set default location

        The default location is set on the ocean grid. Local coordinates
        on (any) other grids are then updated using local host searching.

        Parameters
        ----------
        particle: *Particle
            The particle.
        """
        self.ocean_data_reader.set_default_location(particle)

        if self.using_atmosphere_data:
            self.atmosphere_data_reader.find_host_using_local_search(particle)

        if self.using_waves_data:
            self.waves_data_reader.find_host_using_local_search(particle)

    cdef set_local_coordinates(self, Particle *particle):
        """ Set local coordinates

        Parameters
        ----------
        particle: *Particle
            The particle.
        """
        self.ocean_data_reader.set_local_coordinates(particle)

        if self.using_atmosphere_data:
            self.atmosphere_data_reader.set_local_coordinates(particle)

        if self.using_waves_data:
            self.waves_data_reader.set_local_coordinates(particle)

    cdef DTYPE_INT_t set_vertical_grid_vars(self, DTYPE_FLOAT_t time,
                                            Particle *particle) except INT_ERR:
        """ Set variables that locate the particle within the vertical grid

        Parameters
        ----------
        time: float
            Current time.

        particle: *Particle
            The particle.
        """
        return self.ocean_data_reader.set_vertical_grid_vars(time, particle)

    cpdef DTYPE_FLOAT_t get_xmin(self) except FLOAT_ERR:
        return self.ocean_data_reader.get_xmin()

    cpdef DTYPE_FLOAT_t get_ymin(self) except FLOAT_ERR:
        return self.ocean_data_reader.get_ymin()

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        """ Return the ocean depth from the ocean model

        Parameters:
        -----------
        time : float
            Time.

        particle: *Particle
            Pointer to a Particle object.

        Returns:
        --------
        zmin : float
            The bottom depth.
        """
        return self.ocean_data_reader.get_zmin(time, particle)

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        """ Returns the sea surface height from the ocean model

        Parameters:
        -----------
        time : float
            Time.

        particle: *Particle
            Pointer to a Particle object.
        
        Returns:
        --------
        zmax : float
            Sea surface elevation.
        """
        return self.ocean_data_reader.get_zmax(time, particle)

    cdef void get_velocity(self, DTYPE_FLOAT_t time, Particle* particle,
            DTYPE_FLOAT_t vel[3]) except +:
        """ Return the ocean velocity vector u(t,x,y,z)

        Parameters
        ----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.
        """
        self.ocean_data_reader.get_velocity(time, particle, vel)

    cdef DTYPE_FLOAT_t get_environmental_variable(self, var_name,
            DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        """ Returns the value of the given environmental variable

        Only variables from the ocean model will be returned.

        Parameters:
        -----------
        var_name : str
            The name of the variable. See above for a list of supported options.

        time : float
            Time at which to interpolate.

        particle: *Particle
            Pointer to a Particle object.

        Returns:
        --------
        var : float
            The value of the variable at the particle's position.
        """
        return self.ocean_data_reader.get_environmental_variable(var_name, time,
                particle)

    cdef DTYPE_FLOAT_t get_horizontal_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            Particle* particle) except FLOAT_ERR:
        """ Returns the horizontal eddy diffusivity

        Parameters:
        -----------
        time : float
            Time at which to interpolate.

        particle: *Particle
            Pointer to a Particle object.

        Returns:
        --------
        Ah : float
            The interpolated value of the horizontal eddy diffusivity.
        """
        return self.ocean_data_reader.get_horizontal_eddy_diffusivity(time,
                particle)

    cdef void get_horizontal_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t time,
            Particle* particle, DTYPE_FLOAT_t Ah_prime[2]) except +:
        """ Returns the gradient in the horizontal eddy diffusivity

        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.

        Ah_prime : C array, float
            dAh_dx and dH_dy components stored in a C array of length two.  
        """
        self.ocean_data_reader.get_horizontal_eddy_diffusivity_derivative(time,
                particle, Ah_prime)

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            Particle* particle) except FLOAT_ERR:
        """ Returns the vertical eddy diffusivity
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.
        
        Returns:
        --------
        Kz : float
            The vertical eddy diffusivity.        
        
        """
        return self.ocean_data_reader.get_vertical_eddy_diffusivity(time,
                particle)

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self,
            DTYPE_FLOAT_t time, Particle* particle) except FLOAT_ERR:
        """ Returns the gradient in the vertical eddy diffusivity.

        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.
        
        Returns:
        --------
        k_prime : float
            Gradient in the vertical eddy diffusivity field.
        """
        return self.ocean_data_reader.get_vertical_eddy_diffusivity_derivative(
                time, particle)

    cdef DTYPE_INT_t is_wet(self, DTYPE_FLOAT_t time, Particle *particle) except INT_ERR:
        """ Return an integer indicating whether `host' is wet or dry
        
        Parameters:
        -----------
        time : float
            Time.

        host : int
            Integer that identifies the host element in question
        """
        return self.ocean_data_reader.is_wet(time, particle)
