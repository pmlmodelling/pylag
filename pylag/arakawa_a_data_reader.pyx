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

from libcpp.vector cimport vector

# PyLag cython imports
from particle cimport Particle
from particle_cpp_wrapper cimport to_string
from pylag.data_reader cimport DataReader
from pylag.unstructured cimport UnstructuredGrid
cimport pylag.interpolation as interp
from pylag.math cimport int_min, float_min, get_intersection_point
from pylag.math cimport Intersection

# PyLag python imports
from pylag import variable_library
from pylag.numerics import get_time_direction

cdef class ArakawaADataReader(DataReader):
    """ DataReader for inputs defined on a Arakawa-a grid
    
    Objects of type ArakawaADataReader are intended to manage all access to
    data objects defined on a Arakawa-a grid, including data describing the
    model grid itself as well as model output variables. Provided are methods
    for searching the model grid for host horizontal elements and for
    interpolating gridded field data to a given point in space and time.
    
    Parameters:
    -----------
    config : SafeConfigParser
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
    cdef UnstructuredGrid _unstructured_grid

    # Grid dimensions
    cdef DTYPE_INT_t _n_longitude, _n_latitude, _n_depth, _n_elems, _n_nodes
    
    # Element connectivity
    cdef DTYPE_INT_t[:,:] _nv
    
    # Element adjacency
    cdef DTYPE_INT_t[:,:] _nbe
    
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

    # Full depth mask (1 - sea point, 0 - land point)
    cdef DTYPE_INT_t[:, :] _depth_mask_last
    cdef DTYPE_INT_t[:, :] _depth_mask_next

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

    # Flags that identify whether a given variable should be read in
    cdef bint _has_w, _has_Kh, _has_Ah, _has_is_wet,  _has_zeta

    def __init__(self, config, mediator):
        self.config = config
        self.mediator = mediator

        # Time direction
        self._time_direction = <int>get_time_direction(config)

        # Set flags from config
        self._has_w = self.config.getboolean("OCEAN_CIRCULATION_MODEL", "has_w")
        self._has_Kh = self.config.getboolean("OCEAN_CIRCULATION_MODEL", "has_Kh")
        self._has_Ah = self.config.getboolean("OCEAN_CIRCULATION_MODEL", "has_Ah")
        self._has_is_wet = self.config.getboolean("OCEAN_CIRCULATION_MODEL", "has_is_wet")
        self._has_zeta = self.config.getboolean("OCEAN_CIRCULATION_MODEL", "has_zeta")

        # Check to see if any environmental variables are being saved.
        try:
            env_var_names = self.config.get("OUTPUT", "environmental_variables").strip().split(',')
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            env_var_names = []

        self.env_var_names = []
        for env_var_name in env_var_names:
            env_var_name = env_var_name.strip()
            if env_var_name is not None:
                if env_var_name in variable_library.standard_variable_names.keys():
                    self.env_var_names.append(env_var_name)
                else:
                    raise ValueError('Received unsupported variable {}'.format(env_var_name))

        self._read_grid()

        self._read_time_dependent_vars()

    cpdef setup_data_access(self, start_datetime, end_datetime):
        """ Set up access to time-dependent variables.
        
        Parameters:
        -----------
        start_datetime : Datetime
            Datetime object corresponding to the simulation start time.
        
        end_datetime : Datetime
            Datetime object corresponding to the simulation end time.
        """
        self.mediator.setup_data_access(start_datetime, end_datetime)

        self._read_time_dependent_vars()

    cpdef read_data(self, DTYPE_FLOAT_t time):
        """ Read in time dependent variable data from file?
        
        `time' is used to test if new data should be read in from file. If this
        is the case, arrays containing time-dependent variable data are updated.
        
        Parameters:
        -----------
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
        
        Parameters:
        -----------       
        particle_old: *Particle
            The particle at its old position.

        particle_new: *Particle
            The particle at its new position. The host element will be updated.
        
        Returns:
        --------
        flag : int
            Integer flag that indicates whether or not the seach was successful.
        """
        cdef DTYPE_INT_t flag, host
        
        flag = self._unstructured_grid.find_host_using_local_search(particle_new,
                                                                    particle_old.host_horizontal_elem)
        
        if flag != IN_DOMAIN:
            # Local search failed to find the particle. Perform check to see if
            # the particle has indeed left the model domain
            flag = self._unstructured_grid.find_host_using_particle_tracing(particle_old,
                                                                            particle_new)

        return flag

    cdef DTYPE_INT_t find_host_using_local_search(self, Particle *particle,
                                                  DTYPE_INT_t first_guess) except INT_ERR:
        """ Returns the host horizontal element through local searching.
        
        This function is a wrapper for the same function implemented in UnstructuredGrid.

        Parameters:
        -----------
        particle: *Particle
            The particle.

        DTYPE_INT_t: first_guess
            The first element to start searching.
        
        Returns:
        --------
        flag : int
            Integer flag that indicates whether or not the seach was successful.
        """
        return self._unstructured_grid.find_host_using_local_search(particle, first_guess)

    cdef DTYPE_INT_t find_host_using_global_search(self, Particle *particle) except INT_ERR:
        """ Returns the host horizontal element through global searching.
        
        This function is a wrapper for the same function implemented in UnstructuredGrid.

        Parameters:
        -----------
        particle_old: *Particle
            The particle.
        
        Returns:
        --------
        flag : int
            Integer flag that indicates whether or not the seach was successful.
        """
        return self._unstructured_grid.find_host_using_global_search(particle)

    cdef Intersection get_boundary_intersection(self, Particle *particle_old, Particle *particle_new):
        """ Find the boundary intersection point

        This function is a wrapper for the same function implemented in UnstructuredGrid.

        Parameters:
        -----------
        particle_old: *Particle
            The particle at its old position.

        particle_new: *Particle
            The particle at its new position.

        Returns:
        --------
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

        Parameters:
        -----------
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

        cdef DTYPE_INT_t k

        # Loop over all levels to find the host z layer
        for k in xrange(self._n_depth - 1):
            depth_upper_level = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k)
            depth_lower_level = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k+1)

            if particle.x3 <= depth_upper_level and particle.x3 >= depth_lower_level:
                # Host layer found
                particle.k_layer = k

                # Set the sigma level interpolation coefficient
                particle.omega_interfaces = interp.get_linear_fraction(particle.x3, depth_lower_level, depth_upper_level)

                # Check to see if any of the nodes on each level are masked
                mask_upper_level = self._interp_mask_status_on_level(particle.host_horizontal_elem, k)
                mask_lower_level = self._interp_mask_status_on_level(particle.host_horizontal_elem, k+1)

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
        depth_upper_level = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k)
        if particle.x3 <= zeta and particle.x3 >= depth_lower_level:
            mask_upper_level = self._interp_mask_status_on_level(particle.host_horizontal_elem, k)
            if mask_upper_level == 0:
                particle.k_layer = k
                particle.set_in_vertical_boundary_layer(True)

                return IN_DOMAIN
            else:
                raise RuntimeError('The particle sits below zeta and above the top level, which is masked. Why should it be masked?')

        # Particle is above h but below the lowest depth level
        k = self._n_depth - 1
        depth_upper_level = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, k)
        if particle.x3 >= h and particle.x3 <= depth_upper_level:
            mask_upper_level = self._interp_mask_status_on_level(particle.host_horizontal_elem, k)
            if mask_upper_level == 0:
                particle.k_layer = k
                particle.set_in_vertical_boundary_layer(True)

                return IN_DOMAIN
            else:
                raise RuntimeError('The particle sits above h but below the bottom level, which is masked. Why should it be masked?')

        # Particle is outside the vertical grid
        return BDY_ERROR

    cpdef DTYPE_FLOAT_t get_xmin(self) except FLOAT_ERR:
        return self._xmin

    cpdef DTYPE_FLOAT_t get_ymin(self) except FLOAT_ERR:
        return self._ymin

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        """ Returns the bottom depth in cartesian coordinates

        h is defined at element nodes. Linear interpolation in space is used
        to compute h(x,y). NB the negative of h (which is +ve downwards) is
        returned.

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
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier  
        cdef vector[DTYPE_FLOAT_t] h_tri = vector[DTYPE_FLOAT_t](N_VERTICES, -999.) # Bathymetry at nodes
        cdef DTYPE_FLOAT_t h # Bathymetry at (x1, x2)

        for i in xrange(N_VERTICES):
            vertex = self._nv[i,particle.host_horizontal_elem]
            h_tri[i] = self._h[vertex]

        h = interp.interpolate_within_element(h_tri, particle.get_phi())

        return -h

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        """ Returns the sea surface height in cartesian coordinates

        zeta is defined at element nodes. Interpolation proceeds through linear
        interpolation in time followed by interpolation in space.

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
        pass
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier
        cdef DTYPE_FLOAT_t zeta # Sea surface elevation at (t, x1, x2)

        # Intermediate arrays
        cdef DTYPE_FLOAT_t zeta_last
        cdef DTYPE_FLOAT_t zeta_next
        cdef vector[DTYPE_FLOAT_t] zeta_tri = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)

        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        for i in xrange(N_VERTICES):
            vertex = self._nv[i, particle.host_horizontal_elem]
            zeta_last = self._zeta_last[vertex]
            zeta_next = self._zeta_next[vertex]

            # Interpolate in time
            if zeta_last == zeta_next:
                zeta_tri[i] = zeta_last
            else:
                zeta_tri[i] = interp.linear_interp(time_fraction, zeta_last, zeta_next)

        # Interpolate in space
        zeta = interp.interpolate_within_element(zeta_tri, particle.get_phi())

        return zeta

    cdef get_velocity(self, DTYPE_FLOAT_t time, Particle* particle,
            DTYPE_FLOAT_t vel[3]):
        """ Returns the velocity u(t,x,y,z) through linear interpolation
        
        Returns the velocity u(t,x,y,z) through interpolation for a particle.

        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.

        Return:
        -------
        vel : C array, float
            u/v/w velocity components stored in a C array.           
        """
        vel[0] = self._get_variable(self._u_last, self._u_next, time, particle)
        vel[1] = self._get_variable(self._v_last, self._v_next, time, particle)
        if self._has_w:
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

        Parameters:
        -----------
        time : float
            Time at which to interpolate.

        particle: *Particle
            Pointer to a Particle object.

        Returns:
        --------
        viscofh : float
            The interpolated value of the horizontal eddy viscosity at the specified point in time and space.
        """
        cdef DTYPE_FLOAT_t var  # ah at (t, x1, x2, x3)

        var = self._get_variable(self._ah_last, self._ah_next, time, particle)

        return var

    cdef get_horizontal_eddy_viscosity_derivative(self, DTYPE_FLOAT_t time,
            Particle* particle, DTYPE_FLOAT_t Ah_prime[2]):
        """ Returns the gradient in the horizontal eddy viscosity

        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.

        Ah_prime : C array, float
            dAh_dx and dH_dy components stored in a C array of length two.  

        TODO:
        -----
        1) Test this.

        References:
        -----------
        Lynch, D. R. et al (2014). Particles in the coastal ocean: theory and
        applications. Cambridge: Cambridge University Press.
        doi.org/10.1017/CBO9781107449336
        """
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier

        # Variables used in interpolation in time
        cdef DTYPE_FLOAT_t time_fraction

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
        self._unstructured_grid.get_grad_phi(particle.host_horizontal_elem, dphi_dx, dphi_dy)

        # No vertical interpolation for particles near to the bottom,
        if particle.get_in_vertical_boundary_layer() is True:
            # Extract ah near to the boundary
            for i in xrange(N_VERTICES):
                vertex = self._nv[i, particle.host_horizontal_elem]
                ah_tri_t_last_level_1[i] = self._ah_last[particle.k_layer, vertex]
                ah_tri_t_next_level_1[i] = self._ah_next[particle.k_layer, vertex]

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
                vertex = self._nv[i,particle.host_horizontal_elem]
                ah_tri_t_last_level_1[i] = self._ah_last[particle.k_layer+1, vertex]
                ah_tri_t_next_level_1[i] = self._ah_next[particle.k_layer+1, vertex]
                ah_tri_t_last_level_2[i] = self._ah_last[particle.k_layer, vertex]
                ah_tri_t_next_level_2[i] = self._ah_next[particle.k_layer, vertex]

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
            Ah_prime[0] = interp.linear_interp(particle.omega_interfaces, dah_dx_level_1, dah_dx_level_2)
            Ah_prime[1] = interp.linear_interp(particle.omega_interfaces, dah_dy_level_1, dah_dy_level_2)

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            Particle* particle) except FLOAT_ERR:
        """ Returns the vertical eddy diffusivity through linear interpolation.
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.
        
        Returns:
        --------
        kh : float
            The vertical eddy diffusivity.        
        
        """
        cdef DTYPE_FLOAT_t var  # kh at (t, x1, x2, x3)

        var = self._get_variable(self._kh_last, self._kh_next, time, particle)

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

        TODO:
        -----
        1) Can this be done using a specialised interpolating object? Would need to
        compute depths and kh at all levels, then interpolate between then. Would
        need to give some thought to how to do this efficiently.
        2) Test this.

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
        cdef DTYPE_FLOAT_t kh_0, kh_1, kh_2, kh_3
        cdef DTYPE_FLOAT_t z_0, z_1, z_2, z_3
        cdef DTYPE_FLOAT_t dkh_lower_level, dkh_upper_level

        if particle.get_in_vertical_boundary_layer() == True:
            kh_0 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, particle.k_layer-1)
            z_0 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, particle.k_layer-1)

            kh_1 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, particle.k_layer)
            z_1 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, particle.k_layer)

            dkh_upper_level = (kh_0 - kh_1) / (z_0 - z_1)

            return dkh_upper_level

        if particle.k_layer == 0:
            kh_0 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, particle.k_layer)
            z_0 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, particle.k_layer)

            kh_1 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, particle.k_layer+1)
            z_1 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, particle.k_layer+1)

            kh_2 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, particle.k_layer+2)
            z_2 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, particle.k_layer+2)

            dkh_lower_level = (kh_0 - kh_2) / (z_0 - z_2)
            dkh_upper_level = (kh_0 - kh_1) / (z_0 - z_1)

        elif particle.k_layer == self._n_depth - 2:
            kh_0 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, particle.k_layer-1)
            z_0 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, particle.k_layer-1)

            kh_1 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, particle.k_layer)
            z_1 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, particle.k_layer)

            kh_2 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, particle.k_layer+1)
            z_2 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, particle.k_layer+1)

            dkh_lower_level = (kh_1 - kh_2) / (z_1 - z_2)
            dkh_upper_level = (kh_0 - kh_2) / (z_0 - z_2)

        else:
            kh_0 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, particle.k_layer-1)
            z_0 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, particle.k_layer-1)

            kh_1 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, particle.k_layer)
            z_1 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, particle.k_layer)

            kh_2 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, particle.k_layer+1)
            z_2 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, particle.k_layer+1)

            kh_3 = self._get_variable_on_level(self._kh_last, self._kh_next, time, particle, particle.k_layer+2)
            z_3 = self._get_variable_on_level(self._depth_levels_last, self._depth_levels_next, time, particle, particle.k_layer+2)

            dkh_lower_level = (kh_1 - kh_3) / (z_1 - z_3)
            dkh_upper_level = (kh_0 - kh_2) / (z_0 - z_2)

        return interp.linear_interp(particle.omega_interfaces, dkh_lower_level, dkh_upper_level)

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
        
        Parameters:
        -----------
        time : float
            Time (unused)

        host : int
            Integer that identifies the host element in question
        """
        cdef DTYPE_FLOAT_t zmin_last, zmax_last
        cdef DTYPE_FLOAT_t zmin_mext, zmax_next

        zmin_last = self.get_zmin(self._time_last, particle)
        zmax_last = self.get_zmax(self._time_last, particle)
        zmin_next = self.get_zmin(self._time_next, particle)
        zmax_next = self.get_zmax(self._time_next, particle)

        if self._wet_cells_last[particle.host_horizontal_elem] == 0 or self._wet_cells_next[particle.host_horizontal_elem] == 0:
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

        Parameters:
        -----------
        var_last : 2D MemoryView
            Array of variable values at the last time index.

        var_next : 2D MemoryView
            Array of variable values at the next time index.

        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object. 
        
        Returns:
        --------
        var : float
            The interpolated value of the variable at the specified point in time and space.
        """
        # Interpolated values on lower and upper bounding depth levels
        cdef DTYPE_FLOAT_t var_level_1
        cdef DTYPE_FLOAT_t var_level_2

        # No vertical interpolation for particles near to the bottom
        if particle.get_in_vertical_boundary_layer() is True:
            return self._get_variable_on_level(var_last, var_next, time, particle, particle.k_layer)
        else:
            var_level_1 = self._get_variable_on_level(var_last, var_next, time, particle, particle.k_layer+1)
            var_level_2 = self._get_variable_on_level(var_last, var_next, time, particle, particle.k_layer)

            return interp.linear_interp(particle.omega_interfaces, var_level_1, var_level_2)

    cdef DTYPE_FLOAT_t _get_variable_on_level(self, DTYPE_FLOAT_t[:, :] var_last_arr, DTYPE_FLOAT_t[:, :] var_next_arr,
            DTYPE_FLOAT_t time, Particle* particle, DTYPE_INT_t k_level) except FLOAT_ERR:
        """ Returns the value of the variable on a level through linear interpolation

        Private method for interpolating fields specified at element nodes on depth levels.
        For particle at depths above h and above a lower level with masked nodes, extrapolation
        is used.

        Parameters:
        -----------
        var_last_arr : 2D MemoryView
            Array of variable values at the last time index.

        var_next_arr : 2D MemoryView
            Array of variable values at the next time index.

        time : float
            Time at which to interpolate.

        particle: *Particle
            Pointer to a Particle object.

        k_level : int
            The dpeth level on which to interpolate.

        Returns:
        --------
        var : float
            The interpolated value of the variable on the specified level
        """
        cdef int vertex # Vertex identifier
        cdef DTYPE_FLOAT_t time_fraction
        cdef DTYPE_FLOAT_t var_last, var_next
        cdef vector[DTYPE_FLOAT_t] var_nodes = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)
        cdef DTYPE_FLOAT_t var
        cdef DTYPE_INT_t i

        # Time fraction
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        for i in xrange(N_VERTICES):
            vertex = self._nv[i, particle.host_horizontal_elem]
            var_last = var_last_arr[k_level, vertex]
            var_next = var_next_arr[k_level, vertex]

            if var_last != var_next:
                var_nodes[i] = interp.linear_interp(time_fraction, var_last, var_next)
            else:
                var_nodes[i] = var_last

        var = interp.interpolate_within_element(var_nodes, particle.get_phi())

        return var

    def _read_grid(self):
        """ Set grid and coordinate variables.
        
        All communications go via the mediator in order to guarantee support for
        both serial and parallel simulations.
        
        Parameters:
        -----------
        N/A
        
        Returns:
        --------
        N/A
        """
        # Read in the grid's dimensions
        self._n_longitude = self.mediator.get_dimension_variable('longitude')
        self._n_latitude = self.mediator.get_dimension_variable('latitude')
        self._n_depth = self.mediator.get_dimension_variable('depth')
        self._n_nodes = self.mediator.get_dimension_variable('node')
        self._n_elems = self.mediator.get_dimension_variable('element')

        # Grid connectivity/adjacency
        self._nv = self.mediator.get_grid_variable('nv', (3, self._n_elems), DTYPE_INT)
        self._nbe = self.mediator.get_grid_variable('nbe', (3, self._n_elems), DTYPE_INT)

        # Raw grid x/y or lat/lon coordinates
        coordinate_system = self.config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system").strip().lower()

        if coordinate_system == "spherical":
            x = self.mediator.get_grid_variable('longitude', (self._n_nodes), DTYPE_FLOAT)
            y = self.mediator.get_grid_variable('latitude', (self._n_nodes), DTYPE_FLOAT)
            xc = self.mediator.get_grid_variable('longitude_c', (self._n_elems), DTYPE_FLOAT)
            yc = self.mediator.get_grid_variable('latitude_c', (self._n_elems), DTYPE_FLOAT)

            # Don't apply offsets in spherical case - set them to 0.0!
            self._xmin = 0.0
            self._ymin = 0.0
        else:
            raise ValueError("Unsupported model coordinate system `{}'".format(coordinate_system))

        # Apply offsets
        x = x - self._xmin
        y = y - self._ymin
        xc = xc - self._xmin
        yc = yc - self._ymin

        # Initialise unstructured grid
        self._unstructured_grid = UnstructuredGrid(self.config, self._n_nodes, self._n_elems, self._nv, self._nbe, x, y, xc, yc)

        # Depth levels at nodal coordinates. Assumes and requires that depth is positive down. The -1 multiplier
        # flips this so that depth is positive up from the zero geoid.
        self._reference_depth_levels = -1.0 * self.mediator.get_grid_variable('depth', (self._n_depth), DTYPE_FLOAT)

        # Bathymetry
        self._h = self.mediator.get_grid_variable('h', (self._n_nodes), DTYPE_FLOAT)

        # Land sea mask
        self._land_sea_mask = self.mediator.get_grid_variable('mask', (self._n_elems), DTYPE_INT)

        # Initialise depth level arrays (but don't fill for now)
        self._depth_levels_last = np.empty((self._n_depth, self._n_nodes), dtype=DTYPE_FLOAT)
        self._depth_levels_next = np.empty((self._n_depth, self._n_nodes), dtype=DTYPE_FLOAT)

        # Initialise is wet arrays (but don't fill for now)
        self._wet_cells_last = np.empty((self._n_elems), dtype=DTYPE_INT)
        self._wet_cells_next = np.empty((self._n_elems), dtype=DTYPE_INT)

    cdef _read_time_dependent_vars(self):
        """ Update time variables and memory views for data fields.
        
        For each time-dependent variable needed by PyLag two references
        are stored. These correspond to the last and next time points at which
        data was saved. Together these bound PyLag's current time point.
        
        All communications go via the mediator in order to guarantee support for
        both serial and parallel simulations.
        
        Parameters:
        -----------
        N/A
        
        Returns:
        --------
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
            zeta_last = self.mediator.get_time_dependent_variable_at_last_time_index('zos', (self._n_latitude, self._n_longitude), DTYPE_FLOAT)
            self._zeta_last = self._reshape_var(zeta_last, ('latitude', 'longitude'))

            zeta_next = self.mediator.get_time_dependent_variable_at_next_time_index('zos', (self._n_latitude, self._n_longitude), DTYPE_FLOAT)
            self._zeta_next = self._reshape_var(zeta_next, ('latitude', 'longitude'))
        else:
            # If zeta wasn't given, set it to zero throughout
            self._zeta_last = np.zeros((self._n_nodes), dtype=DTYPE_FLOAT)
            self._zeta_next = np.zeros((self._n_nodes), dtype=DTYPE_FLOAT)

        # Update memory views for u
        u_last = self.mediator.get_time_dependent_variable_at_last_time_index('uo', (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
        self._u_last = self._reshape_var(u_last, ('depth', 'latitude', 'longitude'))

        u_next = self.mediator.get_time_dependent_variable_at_next_time_index('uo', (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
        self._u_next = self._reshape_var(u_next, ('depth', 'latitude', 'longitude'))

        # Update memory views for v
        v_last = self.mediator.get_time_dependent_variable_at_last_time_index('vo', (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
        self._v_last = self._reshape_var(v_last, ('depth', 'latitude', 'longitude'))

        v_next = self.mediator.get_time_dependent_variable_at_next_time_index('vo', (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
        self._v_next = self._reshape_var(v_next, ('depth', 'latitude', 'longitude'))

        # Update memory views for w
        if self._has_w:
            w_last = self.mediator.get_time_dependent_variable_at_last_time_index('wo', (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
            self._w_last = self._reshape_var(w_last, ('depth', 'latitude', 'longitude'))

            w_next = self.mediator.get_time_dependent_variable_at_next_time_index('wo', (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
            self._w_next = self._reshape_var(w_next, ('depth', 'latitude', 'longitude'))

        # Update depth mask
        depth_mask_last = self.mediator.get_mask_at_last_time_index('uo', (self._n_depth, self._n_latitude, self._n_longitude))
        self._depth_mask_last = self._reshape_var(depth_mask_last, ('depth', 'latitude', 'longitude'))

        depth_mask_next = self.mediator.get_mask_at_next_time_index('uo', (self._n_depth, self._n_latitude, self._n_longitude))
        self._depth_mask_next = self._reshape_var(depth_mask_next, ('depth', 'latitude', 'longitude'))

        # Compute actual depth levels using reference values and zeta
        for k in xrange(self._n_depth):
            for i in xrange(self._n_nodes):
                self._depth_levels_last[k, i] = self._reference_depth_levels[k] + self._zeta_last[i]
                self._depth_levels_next[k, i] = self._reference_depth_levels[k] + self._zeta_next[i]

        # Update memory views for kh
        if self._has_Kh:
            kh_last = self.mediator.get_time_dependent_variable_at_last_time_index('kh', (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
            self._kh_last = self._reshape_var(kh_last, ('depth', 'latitude', 'longitude'))

            kh_next = self.mediator.get_time_dependent_variable_at_next_time_index('kh', (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
            self._kh_next = self._reshape_var(kh_next, ('depth', 'latitude', 'longitude'))

        # Update memory views for viscofh
        if self._has_Ah:
            ah_last = self.mediator.get_time_dependent_variable_at_last_time_index('ah', (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
            self._ah_last = self._reshape_var(ah_last, ('depth', 'latitude', 'longitude'))

            ah_next = self.mediator.get_time_dependent_variable_at_next_time_index('ah', (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
            self._ah_next = self._reshape_var(ah_next, ('depth', 'latitude', 'longitude'))

        # Set is wet status
        # NB the status of cells is inferred from the depth mask and the land-sea element mask. If a surface cell is
        # masked but it is not a land cell, then it is assumed to be dry.
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
            var_name = variable_library.standard_variable_names['thetao']
            thetao_next = self.mediator.get_time_dependent_variable_at_next_time_index(var_name, (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
            self._thetao_next = self._reshape_var(thetao_next, ('depth', 'latitude', 'longitude'))

            thetao_last = self.mediator.get_time_dependent_variable_at_last_time_index(var_name, (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
            self._thetao_last = self._reshape_var(thetao_last, ('depth', 'latitude', 'longitude'))

        if 'so' in self.env_var_names:
            var_name = variable_library.standard_variable_names['so']
            so_next = self.mediator.get_time_dependent_variable_at_next_time_index(var_name, (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
            self._so_next = self._reshape_var(so_next, ('depth', 'latitude', 'longitude'))

            so_last = self.mediator.get_time_dependent_variable_at_last_time_index(var_name, (self._n_depth, self._n_latitude, self._n_longitude), DTYPE_FLOAT)
            self._so_last = self._reshape_var(so_last, ('depth', 'latitude', 'longitude'))

        return

    def _reshape_var(self, var, dimensions):
        """ Reshape variable for PyLag

        Variables with the following dimensions are supported:

        2D - [lat, lon] in any order

        3D - [depth, lat, lon] in any order

        Parameters:
        -----------
        var : NDArray
            The variable to sort

        dimensions : tuple(str, str, ...)
            List of dimensions

        Returns:
        --------
        var_reshaped : NDArray
            Reshaped variable
        """
        n_var_dimensions = len(var.shape)
        n_dimensions = len(dimensions)

        if n_var_dimensions != n_dimensions:
            raise ValueError('Array dimension sizes do not match')

        if n_dimensions == 2:
            lat_index = dimensions.index('latitude')
            lon_index = dimensions.index('longitude')

            # Shift axes to give [x, y]
            var = np.moveaxis(var, lon_index, 0)

            return var.reshape(np.prod(var.shape), order='C')[:]

        elif n_dimensions == 3:
            depth_index = dimensions.index('depth')
            lat_index = dimensions.index('latitude')
            lon_index = dimensions.index('longitude')

            # Shift axes to give [z, x, y]
            var = np.moveaxis(var, depth_index, 0)

            # Update lat/lon indices if needed
            if depth_index > lat_index:
                lat_index += 1
            if depth_index > lon_index:
                lon_index += 1

            var = np.moveaxis(var, lon_index, 1)

            return var.reshape(var.shape[0], np.prod(var.shape[1:]), order='C')[:]
        else:
            raise ValueError('Unsupported number of dimensions {}.'.format(n_dimensions))

    cdef DTYPE_INT_t _interp_mask_status_on_level(self,
            DTYPE_INT_t host, DTYPE_INT_t kidx) except INT_ERR:
        """ Return the masked status of the given depth level
 
        Parameters:
        -----------
        phi : c array, float
            Array of length three giving the barycentric coordinates at which 
            to interpolate.
            
        host : int
            Host element index.

        kidx : int
            Sigma layer on which to interpolate.

        Returns:
        --------
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

