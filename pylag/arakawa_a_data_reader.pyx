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
from particle cimport Particle, to_string
from pylag.data_reader cimport DataReader
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

    # Grid dimensions
    cdef DTYPE_INT_t _n_longitude, _n_latitude, _n_depth, _n_elems, _n_nodes
    
    # Element connectivity
    cdef DTYPE_INT_t[:,:] _nv
    
    # Element adjacency
    cdef DTYPE_INT_t[:,:] _nbe
    
    # Nodal coordinates
    cdef DTYPE_FLOAT_t[:] _x
    cdef DTYPE_FLOAT_t[:] _y

    # Element centre coordinates
    cdef DTYPE_FLOAT_t[:] _xc
    cdef DTYPE_FLOAT_t[:] _yc

    # Minimum nodal x/y values
    cdef DTYPE_FLOAT_t _xmin
    cdef DTYPE_FLOAT_t _ymin
    
    cdef DTYPE_FLOAT_t[:] _depth_levels

    # Bathymetry
    cdef DTYPE_FLOAT_t[:] _h
    
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
    cdef bint _has_w, _has_Kh, _has_Ah, _has_is_wet

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

        # Check to see if any environmental variables are being saved.
        try:
            env_var_names = self.config.get("OUTPUT", "environmental_variables").strip().split(',')
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            env_var_names = []

        self.env_var_names = []
        for env_var_name in env_var_names:
            env_var_name = env_var_name.strip()
            if env_var_name is not None:
                if env_var_name in variable_library.fvcom_variable_names.keys():
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
        
        flag = self.find_host_using_local_search(particle_new,
                                                 particle_old.host_horizontal_elem)
        
        if flag != IN_DOMAIN:
            # Local search failed to find the particle. Perform check to see if
            # the particle has indeed left the model domain
            flag = self.find_host_using_particle_tracing(particle_old,
                                                         particle_new)

        return flag

    cdef DTYPE_INT_t find_host_using_local_search(self, Particle *particle,
                                                  DTYPE_INT_t first_guess) except INT_ERR:
        """ Returns the host horizontal element through local searching.
        
        Use a local search for the host horizontal element in which the next
        element to be search is determined by the barycentric coordinates of
        the last element to be searched.
        
        The function returns a flag that indicates whether or not the particle
        has been found within the domain. If it has, its host element will 
        have been set appropriately. If not, a search error is returned. The
        algorithm cannot reliably detect boundary crossings, so no attempt
        is made to try and flag if a boundary crossing occurred.
        
        We also keep track of the second to last element to be searched in order
        to guard against instances when the model gets stuck alternately testing
        two separate neighbouring elements.
        
        Conventions
        -----------
        flag = IN_DOMAIN:
            This indicates that the particle was found successfully. Host is
            is the index of the new host element.
        
        flag = BDY_ERROR:
            The host element was not found.
        
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
        # Intermediate arrays/variables
        cdef DTYPE_FLOAT_t phi[N_VERTICES]
        cdef DTYPE_FLOAT_t phi_test

        cdef bint host_found

        cdef DTYPE_INT_t n_host_land_boundaries
        
        cdef DTYPE_INT_t flag, guess, last_guess, second_to_last_guess

        # Check for non-sensical start points.
        guess = first_guess
        if guess < 0:
            raise ValueError('Invalid start point for local host element '\
                    'search. Start point = {}'.format(guess))

        host_found = False
        last_guess = -1
        second_to_last_guess = -1
        
        while True:
            # Barycentric coordinates
            self._get_phi(particle.x1, particle.x2, guess, phi)

            # Check to see if the particle is in the current element
            phi_test = float_min(float_min(phi[0], phi[1]), phi[2])
            if phi_test >= 0.0:
                host_found = True

            # If the particle has walked into an element with two land
            # boundaries flag this as an error.
            if host_found is True:
                n_host_land_boundaries = 0
                for i in xrange(3):
                    if self._nbe[i,guess] == -1:
                        n_host_land_boundaries += 1

                if n_host_land_boundaries < 2:
                    # Normal element
                    particle.host_horizontal_elem = guess

                    # Set the particle's local coordiantes
                    for i in xrange(3):
                        particle.phi[i] = phi[i]

                    return IN_DOMAIN
                else:
                    # Element has two land boundaries
                    return BDY_ERROR

            # If not, use phi to select the next element to be searched
            second_to_last_guess = last_guess
            last_guess = guess
            if phi[0] == phi_test:
                guess = self._nbe[0,last_guess]
            elif phi[1] == phi_test:
                guess = self._nbe[1,last_guess]
            else:
                guess = self._nbe[2,last_guess]

            # Check for boundary crossings
            if guess == -1 or guess == -2:
                return BDY_ERROR
            
            # Check that we are not alternately checking the same two elements
            if guess == second_to_last_guess:
                return BDY_ERROR

    cdef DTYPE_INT_t find_host_using_particle_tracing(self, Particle *particle_old,
                                                      Particle *particle_new) except INT_ERR:
        """ Try to find the new host element using the particle's pathline
        
        The algorithm navigates between elements by finding the exit point
        of the pathline from each element. If the pathline terminates within
        a valid host element, the index of the new host element is set and a
        flag indicating that a valid host element was successfully found is
        returned. If the pathline crosses a model boundary, the last element the
        host horizontal element of the new particle is set to the last element the
        particle passed through before exiting the domain and a flag indicating
        the type of boundary crossed is returned. Flag conventions are the same
        as those applied in local host element searching.

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
        cdef int i # Loop counter
        cdef int vertex # Vertex identifier
        cdef DTYPE_INT_t elem, last_elem, current_elem # Element identifies
        cdef DTYPE_INT_t flag, host

        # Intermediate arrays/variables
        cdef DTYPE_FLOAT_t x_tri[N_VERTICES]
        cdef DTYPE_FLOAT_t y_tri[N_VERTICES]

        # 2D position vectors for the end points of the element's side
        cdef DTYPE_FLOAT_t x1[2]
        cdef DTYPE_FLOAT_t x2[2]
        
        # 2D position vectors for the particle's previous and new position
        cdef DTYPE_FLOAT_t x3[2]
        cdef DTYPE_FLOAT_t x4[2]
        
        # 2D position vector for the intersection point
        cdef DTYPE_FLOAT_t xi[2]

        # Intermediate arrays
        cdef DTYPE_INT_t x1_indices[3]
        cdef DTYPE_INT_t x2_indices[3]
        cdef DTYPE_INT_t nbe_indices[3]
        
        x1_indices = [0,1,2]
        x2_indices = [1,2,0]
        nbe_indices = [2,0,1]

        # Array indices
        cdef int x1_idx
        cdef int x2_idx
        cdef int nbe_idx

        # Construct arrays to hold the coordinates of the particle's previous
        # position vector and its new position vector
        x3[0] = particle_old.x1; x3[1] = particle_old.x2
        x4[0] = particle_new.x1; x4[1] = particle_new.x2

        # Start the search using the host known to contain (x1_old, x2_old)
        elem = particle_old.host_horizontal_elem

        # Set last_elem equal to elem in the first instance
        last_elem = elem

        while True:
            # Extract nodal coordinates
            for i in xrange(3):
                vertex = self._nv[i,elem]
                x_tri[i] = self._x[vertex]
                y_tri[i] = self._y[vertex]

            # This keeps track of the element currently being checked
            current_elem = elem

            # Loop over all sides of the element to find the land boundary the element crossed
            for i in xrange(3):
                x1_idx = x1_indices[i]
                x2_idx = x2_indices[i]
                nbe_idx = nbe_indices[i]

                # Test to avoid checking the side the pathline just crossed
                if last_elem == self._nbe[nbe_idx, elem]:
                    continue
            
                # End coordinates for the side
                x1[0] = x_tri[x1_idx]; x1[1] = y_tri[x1_idx]
                x2[0] = x_tri[x2_idx]; x2[1] = y_tri[x2_idx]
                
                try:
                    get_intersection_point(x1, x2, x3, x4, xi)
                except ValueError:
                    # Lines do not intersect - check the next one
                    continue

                # Intersection found - keep a record of the last element checked
                last_elem = elem

                # Index for the neighbour element
                elem = self._nbe[nbe_idx, elem]

                # Check to see if the pathline has exited the domain
                if elem >= 0:
                    # Treat elements with two boundaries as land (i.e. set
                    # `flag' equal to -1) and return the last element checked
                    n_host_boundaries = 0
                    for i in xrange(3):
                        if self._nbe[i,elem] == -1:
                            n_host_boundaries += 1
                    if n_host_boundaries == 2:
                        flag = LAND_BDY_CROSSED

                        # Set host to the last element the particle passed through
                        particle_new.host_horizontal_elem = last_elem
                        return flag
                    else:
                        # Intersection found but the pathline has not exited the
                        # domain
                        break
                else:
                    # Particle has crossed a boundary
                    if elem == -1:
                        # Land boundary crossed
                        flag = LAND_BDY_CROSSED
                    elif elem == -2:
                        # Open ocean boundary crossed
                        flag = OPEN_BDY_CROSSED

                    # Set host to the last element the particle passed through
                    particle_new.host_horizontal_elem = last_elem

                    return flag

            if current_elem == elem:
                # Particle has not exited the current element meaning it must
                # still reside in the domain
                flag = IN_DOMAIN
                particle_new.host_horizontal_elem = current_elem
                self.set_local_coordinates(particle_new)

                return flag

    cdef DTYPE_INT_t find_host_using_global_search(self, Particle *particle) except INT_ERR:
        """ Returns the host horizontal element through global searching.
        
        Sequentially search all elements for the given location. Set the particle
        host element if found.
        
        Parameters:
        -----------
        particle_old: *Particle
            The particle.
        
        Returns:
        --------
        flag : int
            Integer flag that indicates whether or not the seach was successful.
        """
        # Intermediate arrays/variables
        cdef DTYPE_FLOAT_t phi[N_VERTICES]
        cdef DTYPE_FLOAT_t phi_test

        cdef bint host_found

        cdef DTYPE_INT_t n_host_land_boundaries

        cdef DTYPE_INT_t i, guess

        host_found = False
        
        for guess in xrange(self._n_elems):
            # Barycentric coordinates
            self._get_phi(particle.x1, particle.x2, guess, phi)

            # Check to see if the particle is in the current element
            phi_test = float_min(float_min(phi[0], phi[1]), phi[2])
            if phi_test >= 0.0:
                host_found = True

            if host_found is True:
                # If the element has two land boundaries, flag the particle as
                # being outside of the domain
                n_host_land_boundaries = 0
                for i in xrange(3):
                    if self._nbe[i,guess] == -1:
                        n_host_land_boundaries += 1

                if n_host_land_boundaries < 2:
                    particle.host_horizontal_elem = guess

                    # Set the particle's local coordiantes
                    for i in xrange(3):
                        particle.phi[i] = phi[i]

                    return IN_DOMAIN
                else:
                    # Element has two land boundaries
                    if self.config.get('GENERAL', 'log_level') == 'DEBUG':
                        logger = logging.getLogger(__name__)
                        logger.warning('Global host element search '
                            'determined that the particle lies within an '
                            'element with two land boundaries. Such elements '
                            'are flagged as lying outside of the model domain.')
                    return BDY_ERROR
        return BDY_ERROR

    cdef Intersection get_boundary_intersection(self, Particle *particle_old, Particle *particle_new):
        """ Find the boundary intersection point

        This function is primarily intended to assist in the application of 
        horizontal boundary conditions where it is often necessary to establish
        the point on a side of an element at which particle crossed before
        exiting the model domain.
        
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
        cdef int i # Loop counter
        cdef int vertex # Vertex identifier

        # Intermediate arrays/variables
        cdef DTYPE_FLOAT_t x_tri[3]
        cdef DTYPE_FLOAT_t y_tri[3]

        # 2D position vectors for the end points of the element's side
        cdef DTYPE_FLOAT_t x1[2]
        cdef DTYPE_FLOAT_t x2[2]
        
        # 2D position vectors for the particle's previous and new position
        cdef DTYPE_FLOAT_t x3[2]
        cdef DTYPE_FLOAT_t x4[2]
        
        # 2D position vector for the intersection point
        cdef DTYPE_FLOAT_t xi[2]

        # Intermediate arrays
        cdef DTYPE_INT_t x1_indices[3]
        cdef DTYPE_INT_t x2_indices[3]
        cdef DTYPE_INT_t nbe_indices[3]

        # Array indices
        cdef int x1_idx
        cdef int x2_idx
        cdef int nbe_idx

        # Variables for computing the number of land boundaries
        cdef DTYPE_INT_t n_land_boundaries
        cdef DTYPE_INT_t nbe

        # The intersection
        cdef Intersection intersection

        intersection = Intersection()

        x1_indices = [0,1,2]
        x2_indices = [1,2,0]
        nbe_indices = [2,0,1]
        
        # Construct arrays to hold the coordinates of the particle's previous
        # position vector and its new position vector
        x3[0] = particle_old.x1; x3[1] = particle_old.x2
        x4[0] = particle_new.x1; x4[1] = particle_new.x2

        # Extract nodal coordinates
        for i in xrange(3):
            vertex = self._nv[i, particle_new.host_horizontal_elem]
            x_tri[i] = self._x[vertex]
            y_tri[i] = self._y[vertex]

        # Loop over all sides of the element to find the land boundary the element crossed
        for i in xrange(3):
            x1_idx = x1_indices[i]
            x2_idx = x2_indices[i]
            nbe_idx = nbe_indices[i]

            nbe = self._nbe[nbe_idx, particle_new.host_horizontal_elem]

            if nbe != -1:
                # Compute the number of land boundaries the neighbour has - elements with two
                # land boundaries are themselves treated as land
                n_land_boundaries = 0
                for i in xrange(3):
                    if self._nbe[i, nbe] == -1:
                        n_land_boundaries += 1

                if n_land_boundaries < 2:
                    continue

            # End coordinates for the side
            x1[0] = x_tri[x1_idx]; x1[1] = y_tri[x1_idx]
            x2[0] = x_tri[x2_idx]; x2[1] = y_tri[x2_idx]

            try:
                get_intersection_point(x1, x2, x3, x4, xi)
                intersection.x1 = x1[0]
                intersection.y1 = x1[1]
                intersection.x2 = x2[0]
                intersection.y2 = x2[1]
                intersection.xi = xi[0]
                intersection.yi = xi[1]
                return intersection
            except ValueError:
                continue

        raise RuntimeError('Failed to calculate boundary intersection.')

    cdef set_default_location(self, Particle *particle):
        """ Set default location

        Move the particle to its host element's centroid.
        """
        particle.x1 = self._xc[particle.host_horizontal_elem]
        particle.x2 = self._yc[particle.host_horizontal_elem]
        self.set_local_coordinates(particle)
        return

    cdef set_local_coordinates(self, Particle *particle):
        """ Set local coordinates
        
        Each particle has associated with it a set of global coordinates
        and a set of local coordinates. Here, the global coordinates and the 
        host horizontal element are used to set the local coordinates.
        
        Parameters:
        -----------
        particle: *Particle
            Pointer to a Particle struct
        """
        cdef DTYPE_FLOAT_t phi[3]
        
        cdef DTYPE_INT_t i
        
        self._get_phi(particle.x1, particle.x2, 
                particle.host_horizontal_elem, phi)
                
        # Check for negative values.
        for i in xrange(3):
            if phi[i] >= 0.0:
                particle.phi[i] = phi[i]
            elif phi[i] >= -EPSILON:
                particle.phi[i] = 0.0
            else:
                print phi[i]
                s = to_string(particle)
                msg = "One or more local coordinates are invalid (phi = {}) \n\n"\
                      "The following information may be used to study the \n"\
                      "failure in more detail. \n\n"\
                      "{}".format(phi[i], s)
                print msg
                
                raise ValueError('One or more local coordinates are negative')

    cdef DTYPE_INT_t set_vertical_grid_vars(self, DTYPE_FLOAT_t time,
                                            Particle *particle) except INT_ERR:
        """ Find the host depth layer
        
        Find the depth layer containing x3. In FVCOM, Sigma levels are counted
        up from 0 starting at the surface, where sigma = 0, and moving downwards
        to the sea floor where sigma = -1. The current sigma layer is
        found by determining the two sigma levels that bound the given z
        position.
        """
        pass

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
        cdef DTYPE_FLOAT_t h_tri[N_VERTICES] # Bathymetry at nodes
        cdef DTYPE_FLOAT_t h # Bathymetry at (x1, x2)

        for i in xrange(N_VERTICES):
            vertex = self._nv[i,particle.host_horizontal_elem]
            h_tri[i] = self._h[vertex]

        h = interp.interpolate_within_element(h_tri, particle.phi)

        return -h

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        """ Returns the sea surface height in cartesian coordinates

        zeta is defined at element nodes. Interpolation proceeds through linear
        interpolation time followed by interpolation in space.

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
        cdef DTYPE_FLOAT_t zeta_tri_t_last[N_VERTICES]
        cdef DTYPE_FLOAT_t zeta_tri_t_next[N_VERTICES]
        cdef DTYPE_FLOAT_t zeta_tri[N_VERTICES]

        for i in xrange(N_VERTICES):
            vertex = self._nv[i,particle.host_horizontal_elem]
            zeta_tri_t_last[i] = self._zeta_last[vertex]
            zeta_tri_t_next[i] = self._zeta_next[vertex]

        # Interpolate in time
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)
        for i in xrange(N_VERTICES):
            zeta_tri[i] = interp.linear_interp(time_fraction, zeta_tri_t_last[i], zeta_tri_t_next[i])

        # Interpolate in space
        zeta = interp.interpolate_within_element(zeta_tri, particle.phi)

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
        pass

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
        pass
#        cdef DTYPE_FLOAT_t var # Environmental variable at (t, x1, x2, x3)
#
#        if var_name in self.env_var_names:
#            if var_name == 'thetao':
#                var = self._get_variable(self._thetao_last, self._thetao_next, time, particle)
#            elif var_name == 'so':
#                var = self._get_variable(self._so_last, self._so_next, time, particle)
#            return var
#        else:
#            raise ValueError("Invalid variable name `{}'".format(var_name))

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
        pass

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

        References:
        -----------
        Lynch, D. R. et al (2014). Particles in the coastal ocean: theory and
        applications. Cambridge: Cambridge University Press.
        doi.org/10.1017/CBO9781107449336
        """
        pass


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
        pass


    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self,
            DTYPE_FLOAT_t time, Particle* particle) except FLOAT_ERR:
        """ Returns the gradient in the vertical eddy diffusivity.
        
        Return a numerical approximation of the gradient in the vertical eddy 
        diffusivity at (t,x,y,z) using central differencing. First, the
        diffusivity is computed on the sigma levels bounding the particle.
        Central differencing is then used to compute the gradient in the
        diffusivity on these levels. Finally, the gradient in the diffusivity
        is interpolated to the particle's exact position. This algorithm
        mirrors that used in GOTMDataReader, which is why it has been implemented
        here. However, in contrast to GOTMDataReader, which calculates the
        gradient in the diffusivity at all levels once each simulation time step,
        resulting in significant time savings, this function is exectued once
        for each particle. It is thus quite costly! To make things worse, the 
        code, as implemented here, is highly repetitive, and no doubt efficiency
        savings could be found. 

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
        pass

    cpdef DTYPE_INT_t is_wet(self, DTYPE_FLOAT_t time, DTYPE_INT_t host) except INT_ERR:
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
        pass
        #if self._has_is_wet:
        #    if self._wet_cells_last[host] == 0 or self._wet_cells_next[host] == 0:
        #        return 0
        #return 1
        
    cdef _get_variable(self, DTYPE_FLOAT_t[:, :] var_last, DTYPE_FLOAT_t[:, :] var_next,
            DTYPE_FLOAT_t time, Particle* particle):
        """ Returns the value of the variable through linear interpolation

        Private method for interpolating fields specified at element nodes on sigma layers.
        This is the case for both viscofh and active and passive tracers. Above and below the
        top and bottom sigma layers respectively values are extrapolated, taking
        a value equal to that at the layer centre. Linear interpolation in the vertical
        is used for z positions lying between the top and bottom sigma layers.
        
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
        pass
#        # No. of vertices and a temporary object used for determining variable
#        # values at the host element's nodes
#        cdef int i # Loop counters
#        cdef int vertex # Vertex identifier
#
#        # Variables used in interpolation in time
#        cdef DTYPE_FLOAT_t time_fraction
#
#        # Intermediate arrays - var
#        cdef DTYPE_FLOAT_t var_tri_t_last_layer_1[N_VERTICES]
#        cdef DTYPE_FLOAT_t var_tri_t_next_layer_1[N_VERTICES]
#        cdef DTYPE_FLOAT_t var_tri_t_last_layer_2[N_VERTICES]
#        cdef DTYPE_FLOAT_t var_tri_t_next_layer_2[N_VERTICES]
#        cdef DTYPE_FLOAT_t var_tri_layer_1[N_VERTICES]
#        cdef DTYPE_FLOAT_t var_tri_layer_2[N_VERTICES]
#
#        # Interpolated values on lower and upper bounding sigma layers
#        cdef DTYPE_FLOAT_t var_layer_1
#        cdef DTYPE_FLOAT_t var_layer_2
#
#        # Time fraction
#        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)
#
#        # No vertical interpolation for particles near to the surface or bottom,
#        # i.e. above or below the top or bottom sigma layer depths respectively.
#        if particle.in_vertical_boundary_layer is True:
#            # Extract values near to the boundary
#            for i in xrange(N_VERTICES):
#                vertex = self._nv[i,particle.host_horizontal_elem]
#                var_tri_t_last_layer_1[i] = var_last[particle.k_layer, vertex]
#                var_tri_t_next_layer_1[i] = var_next[particle.k_layer, vertex]
#
#            # Interpolate in time
#            for i in xrange(N_VERTICES):
#                var_tri_layer_1[i] = interp.linear_interp(time_fraction,
#                                            var_tri_t_last_layer_1[i],
#                                            var_tri_t_next_layer_1[i])
#
#            # Interpolate var within the host element
#            return interp.interpolate_within_element(var_tri_layer_1, particle.phi)
#
#        else:
#            # Extract var on the lower and upper bounding sigma layers
#            for i in xrange(N_VERTICES):
#                vertex = self._nv[i,particle.host_horizontal_elem]
#                var_tri_t_last_layer_1[i] = var_last[particle.k_lower_layer, vertex]
#                var_tri_t_next_layer_1[i] = var_next[particle.k_lower_layer, vertex]
#                var_tri_t_last_layer_2[i] = var_last[particle.k_upper_layer, vertex]
#                var_tri_t_next_layer_2[i] = var_next[particle.k_upper_layer, vertex]
#
#            # Interpolate in time
#            for i in xrange(N_VERTICES):
#                var_tri_layer_1[i] = interp.linear_interp(time_fraction,
#                                            var_tri_t_last_layer_1[i],
#                                            var_tri_t_next_layer_1[i])
#                var_tri_layer_2[i] = interp.linear_interp(time_fraction,
#                                            var_tri_t_last_layer_2[i],
#                                            var_tri_t_next_layer_2[i])
#
#            # Interpolate var within the host element on the upper and lower
#            # bounding sigma layers
#            var_layer_1 = interp.interpolate_within_element(var_tri_layer_1, particle.phi)
#            var_layer_2 = interp.interpolate_within_element(var_tri_layer_2, particle.phi)
#
#            return interp.linear_interp(particle.omega_layers, var_layer_1, var_layer_2)

    cdef DTYPE_FLOAT_t _get_vertical_eddy_diffusivity_on_level(self,
            DTYPE_FLOAT_t time, Particle* particle,
            DTYPE_INT_t k_level) except FLOAT_ERR:
        """ Returns the vertical eddy diffusivity on a level
        
        The vertical eddy diffusivity is defined at element nodes on sigma
        levels. Interpolation is performed first in time, then in x and y to
        give the eddy diffusivity on the specified depth level.
        
        For internal use only.
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        particle : *Particle
            Pointer to a Particle object.
        
        k_level : int
            The dpeth level on which to interpolate.
        
        Returns:
        --------
        kh : float
            The vertical eddy diffusivity at the particle's position on the
            specified level.
        
        """
        pass
#        # Loop counter
#        cdef int i
#
#        # Vertex identifier
#        cdef int vertex
#
#        # Time fraction used for interpolation in time
#        cdef DTYPE_FLOAT_t time_fraction
#
#        # Intermediate arrays - kh
#        cdef DTYPE_FLOAT_t kh_tri_t_last[N_VERTICES]
#        cdef DTYPE_FLOAT_t kh_tri_t_next[N_VERTICES]
#        cdef DTYPE_FLOAT_t kh_tri[N_VERTICES]
#
#        # Interpolated diffusivities on the specified level
#        cdef DTYPE_FLOAT_t kh
#
#        # Extract kh on the lower and upper bounding sigma levels, h and zeta
#        for i in xrange(N_VERTICES):
#            vertex = self._nv[i,particle.host_horizontal_elem]
#            kh_tri_t_last[i] = self._kh_last[k_level, vertex]
#            kh_tri_t_next[i] = self._kh_next[k_level, vertex]
#
#        # Interpolate kh and zeta in time
#        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)
#        for i in xrange(N_VERTICES):
#            kh_tri[i] = interp.linear_interp(time_fraction, kh_tri_t_last[i], kh_tri_t_next[i])
#
#        # Interpolate kh, zeta and h within the host
#        kh = interp.interpolate_within_element(kh_tri, particle.phi)
#
#        return kh

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
        self._x = x - self._xmin
        self._y = y - self._ymin
        self._xc = xc - self._xmin
        self._yc = yc - self._ymin

        # Depth levels at nodal coordinates
        self._depth_levels = self.mediator.get_grid_variable('depth', (self._n_depth), DTYPE_FLOAT)

        # Bathymetry
        self._h = self.mediator.get_grid_variable('h', (self._n_nodes), DTYPE_FLOAT)

    cdef _read_time_dependent_vars(self):
        """ Update time variables and memory views for FVCOM data fields.
        
        For each FVCOM time-dependent variable needed by PyLag two references
        are stored. These correspond to the last and next time points at which
        FVCOM data was saved. Together these bound PyLag's current time point.
        
        All communications go via the mediator in order to guarantee support for
        both serial and parallel simulations.
        
        Parameters:
        -----------
        N/A
        
        Returns:
        --------
        N/A
        """
        # Update time references
        self._time_last = self.mediator.get_time_at_last_time_index()
        self._time_next = self.mediator.get_time_at_next_time_index()

        # Update memory views for zeta
        zeta_last = self.mediator.get_time_dependent_variable_at_last_time_index('zeta', (self._n_latitude, self._n_longitude), DTYPE_FLOAT)
        self._zeta_last = self._reshape_var(zeta_last, ('latitude', 'longitude'))

        zeta_next = self.mediator.get_time_dependent_variable_at_next_time_index('zeta', (self._n_latitude, self._n_longitude), DTYPE_FLOAT)
        self._zeta_next = self._reshape_var(zeta_next, ('latitude', 'longitude'))

        # Update memory views for u, v and w
#        self._u_last = self.mediator.get_time_dependent_variable_at_last_time_index('uo', (self._n_depth_levels, self._n_nodes), DTYPE_FLOAT)
#        self._u_next = self.mediator.get_time_dependent_variable_at_next_time_index('uo', (self._n_depth_levels, self._n_nodes), DTYPE_FLOAT)
#        self._v_last = self.mediator.get_time_dependent_variable_at_last_time_index('vo', (self._n_depth_levels, self._n_nodes), DTYPE_FLOAT)
#        self._v_next = self.mediator.get_time_dependent_variable_at_next_time_index('vo', (self._n_depth_levels, self._n_nodes), DTYPE_FLOAT)

#        if self._has_w:
#            self._w_last = self.mediator.get_time_dependent_variable_at_last_time_index('wo', (self._n_depth_levels, self._n_nodes), DTYPE_FLOAT)
#            self._w_next = self.mediator.get_time_dependent_variable_at_next_time_index('wo', (self._n_depth_levels, self._n_nodes), DTYPE_FLOAT)
#
#        # Update memory views for kh
#        if self._has_Kh:
#            self._kh_last = self.mediator.get_time_dependent_variable_at_last_time_index('kh', (self._n_siglev, self._n_nodes), DTYPE_FLOAT)
#            self._kh_next = self.mediator.get_time_dependent_variable_at_next_time_index('kh', (self._n_siglev, self._n_nodes), DTYPE_FLOAT)
#
#        # Update memory views for viscofh
#        if self._has_Ah:
#            self._viscofh_last = self.mediator.get_time_dependent_variable_at_last_time_index('viscofh', (self._n_siglay, self._n_nodes), DTYPE_FLOAT)
#            self._viscofh_next = self.mediator.get_time_dependent_variable_at_next_time_index('viscofh', (self._n_siglay, self._n_nodes), DTYPE_FLOAT)
#
#        # Update memory views for wet cells
#        if self._has_is_wet:
#            self._wet_cells_last = self.mediator.get_time_dependent_variable_at_last_time_index('wet_cells', (self._n_elems), DTYPE_INT)
#            self._wet_cells_next = self.mediator.get_time_dependent_variable_at_next_time_index('wet_cells', (self._n_elems), DTYPE_INT)
#
#        # Read in data as requested
#        if 'thetao' in self.env_var_names:
#            fvcom_var_name = variable_library.fvcom_variable_names['thetao']
#            self._thetao_last = self.mediator.get_time_dependent_variable_at_last_time_index(fvcom_var_name, (self._n_siglay, self._n_nodes), DTYPE_FLOAT)
#            self._thetao_next = self.mediator.get_time_dependent_variable_at_next_time_index(fvcom_var_name, (self._n_siglay, self._n_nodes), DTYPE_FLOAT)
#
#        if 'so' in self.env_var_names:
#            fvcom_var_name = variable_library.fvcom_variable_names['so']
#            self._so_last = self.mediator.get_time_dependent_variable_at_last_time_index(fvcom_var_name, (self._n_siglay, self._n_nodes), DTYPE_FLOAT)
#            self._so_next = self.mediator.get_time_dependent_variable_at_next_time_index(fvcom_var_name, (self._n_siglay, self._n_nodes), DTYPE_FLOAT)
#
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

    cdef void _get_phi(self, DTYPE_FLOAT_t x1, DTYPE_FLOAT_t x2,
            DTYPE_INT_t host, DTYPE_FLOAT_t phi[N_VERTICES]) except *:
        """ Get barycentric coordinates.
        
        Parameters:
        -----------
        x1 : float
            x-position in cartesian coordinates.
        
        x2 : float
            y-position in cartesian coordinates.
        
        host : int
            Host element.
        
        Returns:
        --------
        phi : C array, float
            Barycentric coordinates.
        """
        
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier

        # Intermediate arrays
        cdef DTYPE_FLOAT_t x_tri[N_VERTICES]
        cdef DTYPE_FLOAT_t y_tri[N_VERTICES]

        for i in xrange(N_VERTICES):
            vertex = self._nv[i,host]
            x_tri[i] = self._x[vertex]
            y_tri[i] = self._y[vertex]

        # Calculate barycentric coordinates
        interp.get_barycentric_coords(x1, x2, x_tri, y_tri, phi)

    cdef void _get_grad_phi(self, DTYPE_INT_t host,
            DTYPE_FLOAT_t dphi_dx[N_VERTICES],
            DTYPE_FLOAT_t dphi_dy[N_VERTICES]) except *:
        """ Get gradient in phi with respect to x and y
        
        Parameters:
        -----------
        host : int
            Host element

        dphi_dx : C array, float
            Gradient with respect to x

        dphi_dy : C array, float
            Gradient with respect to y
        """
        
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier

        # Intermediate arrays
        cdef DTYPE_FLOAT_t x_tri[N_VERTICES]
        cdef DTYPE_FLOAT_t y_tri[N_VERTICES]

        for i in xrange(N_VERTICES):
            vertex = self._nv[i,host]
            x_tri[i] = self._x[vertex]
            y_tri[i] = self._y[vertex]

        # Calculate gradient in barycentric coordinates
        interp.get_barycentric_gradients(x_tri, y_tri, dphi_dx, dphi_dy)

    cdef DTYPE_FLOAT_t _interp_on_sigma_level(self,
            DTYPE_FLOAT_t phi[N_VERTICES], DTYPE_INT_t host,
            DTYPE_INT_t kidx) except FLOAT_ERR:
        """ Return the linearly interpolated value of sigma.
        
        Compute sigma on the specified sigma level within the given host 
        element.
        
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
        sigma: float
            Interpolated value of sigma.
        """
        cdef int vertex # Vertex identifier
        cdef DTYPE_FLOAT_t sigma_nodes[N_VERTICES]
        cdef DTYPE_FLOAT_t sigma # Sigma

        for i in xrange(N_VERTICES):
            vertex = self._nv[i,host]
            sigma_nodes[i] = self._siglev[kidx, vertex]                  

        sigma = interp.interpolate_within_element(sigma_nodes, phi)
        return sigma
