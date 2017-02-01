include "constants.pxi"

import logging

from cpython cimport bool

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cython imports
from pylag.data_reader cimport DataReader
cimport pylag.interpolation as interp
from pylag.math cimport int_min, float_min, get_intersection_point

cdef class FVCOMDataReader(DataReader):
    """ DataReader for FVCOM.
    
    Objects of type FVCOMDataReader are intended to manage all access to FVCOM 
    data objects, including data describing the model grid as well as model
    output variables. Provided are methods for searching the model grid for
    host horizontal elements and for interpolating gridded field data to
    a given point in space and time.
    
    Parameters:
    -----------
    config : SafeConfigParser
        Configuration object.
    
    mediator : Mediator
        Mediator object for managing access to data read from file.
    """
    
    # Configurtion object
    cdef object config

    # Mediator for accessing FVCOM model data read in from file
    cdef object mediator
    
    # Grid dimensions
    cdef DTYPE_INT_t _n_elems, _n_nodes, _n_siglay, _n_siglev
    
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
    
    # Interpolation coefficients
    cdef DTYPE_FLOAT_t[:,:] _a1u
    cdef DTYPE_FLOAT_t[:,:] _a2u
    
    # Sigma layers and levels
    cdef DTYPE_FLOAT_t[:,:] _siglev
    cdef DTYPE_FLOAT_t[:,:] _siglay
    
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
    cdef DTYPE_FLOAT_t[:,:] _omega_last
    cdef DTYPE_FLOAT_t[:,:] _omega_next
    
    # Vertical eddy diffusivities
    cdef DTYPE_FLOAT_t[:,:] _kh_last
    cdef DTYPE_FLOAT_t[:,:] _kh_next
    
    # Horizontal eddy diffusivities
    cdef DTYPE_FLOAT_t[:,:] _viscofh_last
    cdef DTYPE_FLOAT_t[:,:] _viscofh_next
    
    # Time array
    cdef DTYPE_FLOAT_t _time_last
    cdef DTYPE_FLOAT_t _time_next

    # z min/max
    cdef DTYPE_FLOAT_t _zmin
    cdef DTYPE_FLOAT_t _zmax    

    def __init__(self, config, mediator):
        self.config = config
        self.mediator = mediator

        self._read_grid()

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
        time_fraction = interp.get_linear_fraction(time, self._time_last, self._time_next)
        if time_fraction < 0.0 or time_fraction >= 1.0:
            self.mediator.update_reading_frames(time)
            self._read_time_dependent_vars()

    cpdef find_host(self, DTYPE_FLOAT_t xpos_old, DTYPE_FLOAT_t ypos_old,
            DTYPE_FLOAT_t xpos_new, DTYPE_FLOAT_t ypos_new, DTYPE_INT_t guess):
        """ Returns the host horizontal element.
        
        This function first tries to find the new host horizontal element using
        a local search algorithm based on the new point's barycentric
        coordinates. This is relatively fast. However, it can incorrectly flag
        that a particle has left the domain when in-fact it hasn't. For this
        reason, when the local host element search indicates that a particle
        has left the domain, a check is performed based on the particle's
        pathline - if this crosses a known boundary, the particle is deemed
        to have left the domain.

        Two variables are returned. The first is a flag that indicates whether
        or not the particle remains in the domain; the second gives either the
        host element or the last element the particle passed through before
        exiting the domain.
        
        Conventions
        -----------
        flag = 0:
            This indicates that the particle was found successfully. Host is the
            index of the new host element.
        
        flag = -1:
            This indicates that the particle exited the domain across a land
            boundary. Host is set to the last element the particle passed
            through before exiting the domain.

        flag = -2:
            This indicates that the particle exited the domain across an open
            boundary. Host is set to the last element the particle passed
            through before exiting the domain.
        
        Parameters:
        -----------
        xpos_old : float
            Old x-position.

        ypos_old : float
            Old y-position

        xpos_new : float
            New x-position.

        ypos_new : float
            New y-position
        
        guess : int
            First element to try during the search.
        
        Returns:
        --------
        host : int
            ID of the new host horizontal element or the last element the
            particle passed through before exiting the domain.
        """
        cdef DTYPE_INT_t flag, host
        
        flag, host = self.find_host_using_local_search(xpos_new, ypos_new,
                guess)
        
        if flag < 0:
            # Local search failed to find the particle. Perform check to see if
            # the particle has indeed left the model domain
            flag, host = self.find_host_using_particle_tracing(xpos_old,
                    ypos_old, xpos_new, ypos_new, guess)
        
        return flag, host

    cpdef find_host_using_local_search(self, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t first_guess):
        """ Returns the host horizontal element through local searching.
        
        Use a local search for the host horizontal element in which the next
        element to be search is determined by the barycentric coordinates of
        the last.
        
        Two variables are returned. The first is a flag that indicates whether
        or not the search was successful, the second gives either the host
        element or the last element searched before exiting the domain.
        
        Conventions
        -----------
        flag = 0:
            This indicates that the particle was found successfully. Host is
            is the index of the new host element.
        
        flag = -1:
            This indicates that the particle exited the domain across a land
            boundary. Host is set to the last element the particle passed
            through before exiting the domain.

        flag = -2:
            This indicates that the particle exited the domain across an open
            boundary. Host is set to the last element the particle passed
            through before exiting the domain.
        
        Parameters:
        -----------
        xpos : float
            x-position.

        ypos : float
            y-position
        
        first_guess : int
            First element to try during the search.
        
        Returns:
        --------
        flag : int
            Integer flag that indicates whether or not the seach was successful.

        guess : int
            ID of the host horizontal element or the last element searched.
        """
        # Intermediate arrays/variables
        cdef DTYPE_FLOAT_t phi[N_VERTICES]
        cdef DTYPE_FLOAT_t phi_test

        cdef bint host_found

        cdef DTYPE_INT_t n_host_land_boundaries
        
        cdef DTYPE_INT_t flag, last_guess, guess

        # Check for non-sensical start points.
        guess = first_guess
        if guess < 0:
            raise ValueError('Invalid start point for local host element '\
                    'search. Start point = {}'.format(guess))

        host_found = False
        
        while True:
            # Barycentric coordinates
            self._get_phi(xpos, ypos, guess, phi)

            # Check to see if the particle is in the current element
            phi_test = float_min(float_min(phi[0], phi[1]), phi[2])
            if phi_test >= 0.0:
                host_found = True
            elif phi_test >= -EPSILON:
                if self.config.getboolean('GENERAL', 'log_level') == 'DEBUG':
                    logger = logging.getLogger(__name__)
                    logger.warning('EPSILON applied during local host element search.')
                host_found = True

            # If the particle has walked into an element with two land
            # boundaries flag it as having moved outside of the domain - ideally
            # unstructured grids should not include such elements.
            if host_found is True:
                n_host_land_boundaries = 0
                for i in xrange(3):
                    if self._nbe[i,guess] == -1:
                        n_host_land_boundaries += 1

                if n_host_land_boundaries < 2:
                    # Normal element
                    flag = 0
                    return flag, guess
                else:
                    # Element has two land boundaries - mark as land and
                    # return the last element searched
                    if self.config.getboolean('GENERAL', 'log_level') == 'DEBUG':
                        logger = logging.getLogger(__name__)
                        logger.warning('Particle prevented from entering '\
                            'element {} which has two land '\
                            'boundaries.'.format(guess))
                    flag = -1
                return flag, last_guess 

            # If not, use phi to select the next element to be searched
            last_guess = guess
            if phi[0] == phi_test:
                guess = self._nbe[0,last_guess]
            elif phi[1] == phi_test:
                guess = self._nbe[1,last_guess]
            else:
                guess = self._nbe[2,last_guess]

            # Check for boundary crossings
            if guess == -1:
                # Land boundary crossed
                flag = -1
                return flag, last_guess
            elif guess == -2:
                # Open ocean boundary crossed
                flag = -2
                return flag, last_guess

    cpdef find_host_using_particle_tracing(self, DTYPE_FLOAT_t xpos_old,
        DTYPE_FLOAT_t ypos_old, DTYPE_FLOAT_t xpos_new, DTYPE_FLOAT_t ypos_new,
        DTYPE_INT_t last_host):
        """ Try to find the new host element using the particle's pathline
        
        The algorithm navigates between elements by finding the exit point
        of the pathline from each element. If the pathline terminates within
        a valid host element, the index of the new host element is returned
        along with a flag indicating that a valid host element was successfully
        found. If the pathline crosses a model boundary, the last element the
        particle passed through before exiting the domain is returned along
        with a flag indicating the type of boundary crossed. Flag conventions
        are the same as those applied in local host element searching.

        Conventions
        -----------
        flag = 0:
            This indicates that the particle was found successfully. Host is the
            index of the new host element.
        
        flag = -1:
            This indicates that the particle exited the domain across a land
            boundary. Host is set to the last element the particle passed
            through before exiting the domain.

        flag = -2:
            This indicates that the particle exited the domain across an open
            boundary. Host is set to the last element the particle passed
            through before exiting the domain.

        Parameters:
        -----------
        xpos_old : float
            x-position at the last time point.

        ypos_old : float
            y-position at the next time point.

        xpos_new : float
            x-position at the last time point.

        ypos_new : float
            y-position at the next time point.        
        
        last_host : int
            Element to use when computing the particle's barycentric coordinates
        
        Returns:
        --------
        flag : int
            Integer flag that indicates whether or not the seach was successful.

        host : int
            ID of the host horizontal element or the last element searched.
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
        nbe_indices = [1,2,0]
        
        # Construct arrays to hold the coordinates of the particle's previous
        # position vector and its new position vector
        x3[0] = xpos_old; x3[1] = ypos_old
        x4[0] = xpos_new; x4[1] = ypos_new

        # Start the search using the host known to contain (xpos_old, ypos_old)
        elem = last_host
        
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

            # Loop over all sides of the element to check for crossings
            for x1_idx, x2_idx, nbe_idx in zip(x1_indices, x2_indices, nbe_indices):

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
                        flag = -1
                        return flag, last_elem
                    else:
                        # Intersection found but the pathline has not exited the
                        # domain
                        break
                else:
                    # Particle has crossed a boundary
                    if elem == -1:
                        # Land boundary crossed
                        flag = -1
                        return flag, last_elem
                    elif elem == -2:
                        # Open ocean boundary crossed
                        flag = -2
                        return flag, last_elem

            if current_elem == elem:
                # Particle has not exited the current element meaning it must
                # still reside in the domain
                flag = 0
                return flag, current_elem

    cpdef find_host_using_global_search(self, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos):
        """ Returns the host horizontal element through global searching.
        
        Sequentially search all elements for the given location. Return the
        ID of the host horizontal element if it exists and -1 if the particle
        lies outside of the domain.
        
        Parameters:
        -----------
        xpos : float
            x-position.

        ypos : float
            y-position
        
        Returns:
        --------
        host : int
            ID of the host horizontal element.
        """
        # Loop counter
        cdef DTYPE_INT_t i

        # Intermediate arrays/variables
        cdef DTYPE_FLOAT_t phi[N_VERTICES]
        cdef DTYPE_FLOAT_t phi_test
        
        for i in xrange(self._n_elems):
            # Barycentric coordinates
            self._get_phi(xpos, ypos, i, phi)

            # Check to see if the particle is in the current element
            phi_test = float_min(float_min(phi[0], phi[1]), phi[2])
            if phi_test >= 0.0:
                return i
            elif phi_test >= -EPSILON:
                if self.config.get('GENERAL', 'log_level') == 'DEBUG':
                    logger = logging.getLogger(__name__)
                    logger.warning('EPSILON applied during global host element search.')
                return i

        raise ValueError('Point ({}, {}) is not in the model domain.'.format(xpos, ypos))

    cpdef get_boundary_intersection(self, DTYPE_FLOAT_t xpos_old,
        DTYPE_FLOAT_t ypos_old, DTYPE_FLOAT_t xpos_new, DTYPE_FLOAT_t ypos_new,
        DTYPE_INT_t last_host):
        """ Find the boundary intersection point

        This function is primarily intended to assist in the application of 
        horizontal boundary conditions where it is often necessary to establish
        the point on a side of an element at which particle crossed before
        exiting the model domain.
        
        Parameters:
        -----------
        xpos_old : float
            x-position at the last time point.

        ypos_old : float
            y-position at the next time point.

        xpos_new : float
            x-position at the next time point that lies outside of the domain.

        ypos_new : float
            y-position at the next time point that lies outside of the domain.      
        
        last_host : int
            The last element the particle passed through before exiting the
            domain.
        
        Returns:
        --------
        x1, y1, x2, y2, xi, yi :  float
            The end coordinates of the line segment (side) and the intersection
            point.
        """
        cdef int i # Loop counter
        cdef int vertex # Vertex identifier

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
        nbe_indices = [1,2,0]
        
        # Construct arrays to hold the coordinates of the particle's previous
        # position vector and its new position vector
        x3[0] = xpos_old; x3[1] = ypos_old
        x4[0] = xpos_new; x4[1] = ypos_new

        # Extract nodal coordinates
        for i in xrange(3):
            vertex = self._nv[i,last_host]
            x_tri[i] = self._x[vertex]
            y_tri[i] = self._y[vertex]

        # Loop over all sides of the element to check for crossings
        for x1_idx, x2_idx, nbe_idx in zip(x1_indices, x2_indices, nbe_indices):
            # End coordinates for the side
            x1[0] = x_tri[x1_idx]; x1[1] = y_tri[x1_idx]
            x2[0] = x_tri[x2_idx]; x2[1] = y_tri[x2_idx]
            
            try:
                get_intersection_point(x1, x2, x3, x4, xi)
                return x1[0], x1[1], x2[0], x2[1], xi[0], xi[1]
            except ValueError:
                continue
        
        raise RuntimeError('Particle path does not intersect any side of the given element.')

    cpdef DTYPE_INT_t find_zlayer(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
        DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
        DTYPE_INT_t guess):
        """ Find the host vertical layer
        
        Find the host sigma layer. Sigma levels are counted up from 0
        starting at the surface and moving downwards. The current sigma layer is
        found by determining the two sigma levels that bound the given z
        position. Here, `guess' is ignored.
        """
        cdef DTYPE_FLOAT_t phi[N_VERTICES]

        cdef DTYPE_FLOAT_t sigma_upper_level, sigma_lower_level

        cdef DTYPE_INT_t k

        # Compute barycentric coordinates for the given x/y coordinates
        self._get_phi(xpos, ypos, host, phi)

        # Loop over all levels to find the host z layer
        for k in xrange(self._n_siglay):
            sigma_upper_level = self._interp_on_sigma_level(phi, host, k)
            sigma_lower_level = self._interp_on_sigma_level(phi, host, k+1)
            
            if zpos <= sigma_upper_level and zpos >= sigma_lower_level:
                return k
        
        raise ValueError("Particle zpos (={}) not found!".format(zpos))

    cpdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos):
        """ Returns zmin.
        
        In sigma coordinates this is simply -1.0 as detailed in the FVCOM
        manual.

        Parameters:
        -----------
        time : float
            Time.

        xpos : float
            x-position.

        ypos : float
            y-position
        
        Returns:
        --------
        zmin : float
            z min.
        """
        return self._zmin

    cpdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos):
        """ Returns zmax.
        
        In sigma coordinates this is simply 0.0 as detailed in the FVCOM
        manual.

        Parameters:
        -----------
        time : float
            Time.

        xpos : float
            x-position.

        ypos : float
            y-position
        
        Returns:
        --------
        zmax : float
            z max.
        """
        return self._zmax

    cpdef get_bathymetry(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos,
            DTYPE_INT_t host):
        """ Returns the bathymetry through linear interpolation.
        
        h is defined at element nodes. Linear interpolation in space is used
        to compute h(x,y).
        
        Parameters:
        -----------
        xpos : float
            x-position at which to interpolate.
        
        ypos : float
            y-position at which to interpolate.
            
        host : int
            Host horizontal element.

        Return:
        -------
        h : float
            Bathymetry.
        """
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier  
        cdef DTYPE_FLOAT_t phi[N_VERTICES] # Barycentric coordinates 
        cdef DTYPE_FLOAT_t h_tri[N_VERTICES] # Bathymetry at nodes
        cdef DTYPE_FLOAT_t h # Bathymetry at (xpos, ypos)

        # Barycentric coordinates
        self._get_phi(xpos, ypos, host, phi)

        for i in xrange(N_VERTICES):
            vertex = self._nv[i,host]
            h_tri[i] = self._h[vertex]

        h = interp.interpolate_within_element(h_tri, phi)

        return h
    
    cpdef get_sea_sur_elev(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t host):
        """ Returns the sea surface elevation through linear interpolation.
        
        zeta is defined at element nodes. Interpolation proceeds through linear
        interpolation time followed by interpolation in space.
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate.
        
        ypos : float
            y-position at which to interpolate.
            
        host : int
            Host horizontal element.

        Return:
        -------
        zeta : float
            Sea surface elevation.
        """
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier
        cdef DTYPE_FLOAT_t zeta # Sea surface elevation at (t, xpos, ypos)

        # Intermediate arrays
        cdef DTYPE_FLOAT_t zeta_tri_t_last[N_VERTICES]
        cdef DTYPE_FLOAT_t zeta_tri_t_next[N_VERTICES]
        cdef DTYPE_FLOAT_t zeta_tri[N_VERTICES]
        cdef DTYPE_FLOAT_t phi[N_VERTICES]

        # Barycentric coordinates
        self._get_phi(xpos, ypos, host, phi)

        for i in xrange(N_VERTICES):
            vertex = self._nv[i,host]
            zeta_tri_t_last[i] = self._zeta_last[vertex]
            zeta_tri_t_next[i] = self._zeta_next[vertex]

        # Interpolate in time
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)
        for i in xrange(N_VERTICES):
            zeta_tri[i] = interp.linear_interp(time_fraction, zeta_tri_t_last[i], zeta_tri_t_next[i])

        # Interpolate in space
        zeta = interp.interpolate_within_element(zeta_tri, phi)

        return zeta

    cdef get_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_INT_t zlayer, DTYPE_FLOAT_t vel[3]):
        """ Returns the velocity u(t,x,y,z) through linear interpolation
        
        Returns the velocity u(t,x,y,z) through linear interpolation for a 
        particle residing in the horizontal host element `host'. The actual 
        computation is split into two separate parts - one for computing u and 
        v, and one for computing omega. This reflects the fact that u and v are
        defined are element centres on sigma layers, while omega is defined at
        element nodes on sigma levels.

        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate.
        
        ypos : float
            y-position at which to interpolate.

        zpos : float
            z-position at which to interpolate.

        host : int
            Host horizontal element.

        zlayer : int
            Host z layer.

        Return:
        -------
        vel : C array, float
            u/v/w velocity components stored in a C array.           
        """
        # Barycentric coordinates
        cdef DTYPE_FLOAT_t phi[N_VERTICES]
        
        # u/v velocity array
        cdef DTYPE_FLOAT_t vel_uv[2]
        
        cdef DTYPE_INT_t i

        # Barycentric coordinates - precomputed here as required for both u/v 
        # and omega computations
        self._get_phi(xpos, ypos, host, phi)
        
        # Compute u/v velocities and save
        self._get_uv_velocity_using_linear_least_squares_interpolation(time, 
                xpos, ypos, zpos, host, zlayer, phi, vel_uv)
        for i in xrange(2):
            vel[i] = vel_uv[i]
        
        # Compute omega velocity and save
        vel[2] = self._get_omega_velocity(time, xpos, ypos, zpos, host, zlayer, phi)
        return

    cdef get_horizontal_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_INT_t zlayer, DTYPE_FLOAT_t vel[2]):
        """ Returns the u/v velocity components through linear interpolation.
        
        This function is effectively a wrapper for _get_uv_velocity*.
        
        TODO:
        -----
        1) Two schemes are implemented but the choice is hardcoded. Have this
        set by a config parameter?
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate.
        
        ypos : float
            y-position at which to interpolate.

        zpos : float
            z-position at which to interpolate.

        host : int
            Host horizontal element.

        zlayer : int
            Host z layer.

        Return:
        -------
        vel : C array, float
            u/v velocity components stored in a C array.        
        """

        # Barycentric coordinates
        cdef DTYPE_FLOAT_t phi[N_VERTICES]

        self._get_phi(xpos, ypos, host, phi)        
        self._get_uv_velocity_using_linear_least_squares_interpolation(time, 
                xpos, ypos, zpos, host, zlayer, phi, vel)
        return
    
    cdef get_vertical_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_INT_t zlayer):
        """ Returns the vertical velocity through linear interpolation.
        
        This function is effectively a wrapper for _get_vertical_velocity.
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate.
        
        ypos : float
            y-position at which to interpolate.

        zpos : float
            z-position at which to interpolate.

        host : int
            Host horizontal element.

        zlayer : int
            Host z layer.

        Return:
        -------
        omega : float
            Omega velocity.        
        """
        
        # Barycentric coordinates
        cdef DTYPE_FLOAT_t phi[N_VERTICES]

        self._get_phi(xpos, ypos, host, phi)
        return self._get_omega_velocity(time, xpos, ypos, zpos, host, zlayer, phi)

    cpdef get_horizontal_eddy_diffusivity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_INT_t zlayer):
        """ Returns the horizontal eddy diffusivity through linear interpolation.
        
        viscofh is defined at element nodes on sigma layers. Above and below the
        top and bottom sigma layers respectivey viscofh is extrapolated, taking
        a value equal to that on the layer. Linear interpolation in the vertical
        is used for z positions lying between the top and bottom sigma layers.
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate.
        
        ypos : float
            y-position at which to interpolate.

        zpos : float
            z-position at which to interpolate.

        host : int
            Host horizontal element.

        zlayer : int
            Host z layer.        
        
        Returns:
        --------
        viscofh : float
            The horizontal eddy diffusivity.      
        """
        # Barycentric coordinates
        cdef DTYPE_FLOAT_t phi[N_VERTICES]

        # Object describing a point's location within FVCOM's vertical grid. 
        cdef ZGridPosition z_grid_pos

        # No. of vertices and a temporary object used for determining variable
        # values at the host element's nodes
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier

        # Variables used in interpolation in time      
        cdef DTYPE_FLOAT_t time_fraction

        # Variables used in interpolation in z
        cdef DTYPE_FLOAT_t sigma_fraction, sigma_lower_layer, sigma_upper_layer
        
        # Intermediate arrays - viscofh
        cdef DTYPE_FLOAT_t viscofh_tri_t_last_layer_1[N_VERTICES]
        cdef DTYPE_FLOAT_t viscofh_tri_t_next_layer_1[N_VERTICES]
        cdef DTYPE_FLOAT_t viscofh_tri_t_last_layer_2[N_VERTICES]
        cdef DTYPE_FLOAT_t viscofh_tri_t_next_layer_2[N_VERTICES]
        cdef DTYPE_FLOAT_t viscofh_tri_layer_1[N_VERTICES]
        cdef DTYPE_FLOAT_t viscofh_tri_layer_2[N_VERTICES]     
        
        # Interpolated diffusivities on lower and upper bounding sigma layers
        cdef DTYPE_FLOAT_t viscofh_layer_1
        cdef DTYPE_FLOAT_t viscofh_layer_2

        # Barycentric coordinates
        self._get_phi(xpos, ypos, host, phi)
        
        # Set variables describing the position within the vertical grid
        self._get_z_grid_position(zpos, host, zlayer, phi, &z_grid_pos) 

        # Time fraction
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        # No vertical interpolation for particles near to the surface or bottom, 
        # i.e. above or below the top or bottom sigma layer depths respectively.
        if z_grid_pos.in_vertical_boundary_layer is True:
            # Extract viscofh near to the boundary
            for i in xrange(N_VERTICES):
                vertex = self._nv[i,host]
                viscofh_tri_t_last_layer_1[i] = self._viscofh_last[z_grid_pos.k_boundary, vertex]
                viscofh_tri_t_next_layer_1[i] = self._viscofh_next[z_grid_pos.k_boundary, vertex]

            # Interpolate in time
            for i in xrange(N_VERTICES):
                viscofh_tri_layer_1[i] = interp.linear_interp(time_fraction, 
                                            viscofh_tri_t_last_layer_1[i],
                                            viscofh_tri_t_next_layer_1[i])

            # Interpolate viscofh within the host element
            return interp.interpolate_within_element(viscofh_tri_layer_1, phi)

        else:
            # Extract viscofh on the lower and upper bounding sigma layers
            for i in xrange(N_VERTICES):
                vertex = self._nv[i,host]
                viscofh_tri_t_last_layer_1[i] = self._viscofh_last[z_grid_pos.k_lower_layer, vertex]
                viscofh_tri_t_next_layer_1[i] = self._viscofh_next[z_grid_pos.k_lower_layer, vertex]
                viscofh_tri_t_last_layer_2[i] = self._viscofh_last[z_grid_pos.k_upper_layer, vertex]
                viscofh_tri_t_next_layer_2[i] = self._viscofh_next[z_grid_pos.k_upper_layer, vertex]

            # Interpolate in time
            for i in xrange(N_VERTICES):
                viscofh_tri_layer_1[i] = interp.linear_interp(time_fraction, 
                                            viscofh_tri_t_last_layer_1[i],
                                            viscofh_tri_t_next_layer_1[i])
                viscofh_tri_layer_2[i] = interp.linear_interp(time_fraction, 
                                            viscofh_tri_t_last_layer_2[i],
                                            viscofh_tri_t_next_layer_2[i])

            # Interpolate viscofh within the host element on the upper and lower
            # bounding sigma layers
            viscofh_layer_1 = interp.interpolate_within_element(viscofh_tri_layer_1, phi)
            viscofh_layer_2 = interp.interpolate_within_element(viscofh_tri_layer_2, phi)

            # Vertical interpolation
            sigma_lower_layer = self._interp_on_sigma_layer(phi, host, z_grid_pos.k_lower_layer)
            sigma_upper_layer = self._interp_on_sigma_layer(phi, host, z_grid_pos.k_upper_layer)
            sigma_fraction = interp.get_linear_fraction_safe(zpos, sigma_lower_layer, sigma_upper_layer)

            return interp.linear_interp(sigma_fraction, viscofh_layer_1, viscofh_layer_2)

    cpdef get_horizontal_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t time,
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, 
            DTYPE_INT_t host, DTYPE_INT_t zlayer):
        """ NOT YET IMPLEMENTED
        
                Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate.
        
        ypos : float
            y-position at which to interpolate.

        zpos : float
            z-position at which to interpolate.

        host : int
            Host horizontal element.

        zlayer : int
            Host z layer.

        Returns:
        --------
        viscofh_prime : float
            The horizontal eddy diffusivity.
        
        """
        pass

    cpdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos,
            DTYPE_INT_t host, DTYPE_INT_t zlayer):
        """ Returns the vertical eddy diffusivity through linear interpolation.
        
        The vertical eddy diffusivity is defined at element nodes on sigma
        levels. Interpolation is performed first in time, then in x and y,
        and finally in z.
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate.
        
        ypos : float
            y-position at which to interpolate.

        zpos : float
            z-position at which to interpolate.

        host : int
            Host horizontal element.
            
        zlayer : int
            Host z layer.
        
        Returns:
        --------
        kh : float
            The vertical eddy diffusivity.        
        
        """
        # Barycentric coordinates
        cdef DTYPE_FLOAT_t phi[N_VERTICES]

        # Loop counter
        cdef int i
        
        # Vertex identifier
        cdef int vertex
        
        # Time fraction used for interpolation in time        
        cdef DTYPE_FLOAT_t time_fraction

        # Variables used for interpolation in sigma
        cdef DTYPE_FLOAT_t sigma_fraction, sigma_lower_level, sigma_upper_level

        # Intermediate arrays - kh
        cdef DTYPE_FLOAT_t kh_tri_t_last_lower_level[N_VERTICES]
        cdef DTYPE_FLOAT_t kh_tri_t_next_lower_level[N_VERTICES]
        cdef DTYPE_FLOAT_t kh_tri_t_last_upper_level[N_VERTICES]
        cdef DTYPE_FLOAT_t kh_tri_t_next_upper_level[N_VERTICES]
        cdef DTYPE_FLOAT_t kh_tri_lower_level[N_VERTICES]
        cdef DTYPE_FLOAT_t kh_tri_upper_level[N_VERTICES]
        
        # Intermediate arrays - zeta/h
        cdef DTYPE_FLOAT_t zeta_tri_t_last[N_VERTICES]
        cdef DTYPE_FLOAT_t zeta_tri_t_next[N_VERTICES]
        cdef DTYPE_FLOAT_t zeta_tri[N_VERTICES]
        cdef DTYPE_FLOAT_t h_tri[N_VERTICES]        
        
        # Interpolated diffusivities on lower and upper bounding sigma levels
        cdef DTYPE_FLOAT_t kh_lower_level
        cdef DTYPE_FLOAT_t kh_upper_level

        # Interpolated zeta/h
        cdef DTYPE_FLOAT_t zeta
        cdef DTYPE_FLOAT_t h

        # Compute barycentric coordinates for the given x/y coordinates
        self._get_phi(xpos, ypos, host, phi)

        # Extract kh on the lower and upper bounding sigma levels, h and zeta
        for i in xrange(N_VERTICES):
            vertex = self._nv[i,host]
            kh_tri_t_last_lower_level[i] = self._kh_last[zlayer+1, vertex]
            kh_tri_t_next_lower_level[i] = self._kh_next[zlayer+1, vertex]
            kh_tri_t_last_upper_level[i] = self._kh_last[zlayer, vertex]
            kh_tri_t_next_upper_level[i] = self._kh_next[zlayer, vertex]
            zeta_tri_t_last[i] = self._zeta_last[vertex]
            zeta_tri_t_next[i] = self._zeta_next[vertex]
            h_tri[i] = self._h[vertex]

        # Interpolate kh and zeta in time
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)
        for i in xrange(N_VERTICES):
            kh_tri_lower_level[i] = interp.linear_interp(time_fraction, 
                                                kh_tri_t_last_lower_level[i],
                                                kh_tri_t_next_lower_level[i])
            kh_tri_upper_level[i] = interp.linear_interp(time_fraction, 
                                                kh_tri_t_last_upper_level[i],
                                                kh_tri_t_next_upper_level[i])
            zeta_tri[i] = interp.linear_interp(time_fraction, zeta_tri_t_last[i], zeta_tri_t_next[i])

        # Interpolate kh, zeta and h within the host
        kh_lower_level = interp.interpolate_within_element(kh_tri_lower_level, phi)
        kh_upper_level = interp.interpolate_within_element(kh_tri_upper_level, phi)
        zeta = interp.interpolate_within_element(zeta_tri, phi)
        h = interp.interpolate_within_element(h_tri, phi)

        # Interpolate between sigma levels
        sigma_upper_level = self._interp_on_sigma_level(phi, host, zlayer)
        sigma_lower_level = self._interp_on_sigma_level(phi, host, zlayer+1)
        sigma_fraction = interp.get_linear_fraction_safe(zpos, sigma_lower_level, sigma_upper_level)

        return interp.linear_interp(sigma_fraction, kh_lower_level, kh_upper_level) / (h + zeta)**2

    cpdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t time,
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, 
            DTYPE_INT_t host, DTYPE_INT_t zlayer):
        """ Returns the gradient in the vertical eddy diffusivity.
        
        Return a numerical approximation of the gradient in the vertical eddy 
        diffusivity at (t,x,y,z).
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate.
        
        ypos : float
            y-position at which to interpolate.

        zpos : float
            z-position at which to interpolate.

        host : int
            Host horizontal element.

        zlayer : int
            Host z layer.
        
        Returns:
        --------
        k_prime : float
            Gradient in the vertical eddy diffusivity field.
        """
        # Diffusivities
        cdef DTYPE_FLOAT_t kh1, kh2
        
        # Diffusivity gradient
        cdef DTYPE_FLOAT_t k_prime
        
        # Z coordinate vars for the gradient calculation
        cdef DTYPE_FLOAT_t zpos_increment, zpos_incremented
        
        # Z layer for incremented position
        cdef DTYPE_INT_t zlayer_incremented
        
        # Use a point arbitrarily close to zpos (in sigma coordinates) for the 
        # gradient calculation
        zpos_increment = 1.0e-3
        
        # Use the negative of zpos_increment at the top of the water column
        if ((zpos + zpos_increment) > 0.0):
            zpos_increment = -zpos_increment
            
        # A point close to zpos
        zpos_incremented = zpos + zpos_increment
        zlayer_incremented = self.find_zlayer(time, xpos, ypos, zpos_incremented, host, zlayer)

        # Compute the gradient
        kh1 = self.get_vertical_eddy_diffusivity(time, xpos, ypos, zpos, host, zlayer)
        kh2 = self.get_vertical_eddy_diffusivity(time, xpos, ypos, zpos_incremented, host, zlayer_incremented)
        k_prime = (kh2 - kh1) / zpos_increment

        return k_prime

    cdef _get_uv_velocity_using_shepard_interpolation(self, DTYPE_FLOAT_t time,
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, 
            DTYPE_INT_t host, DTYPE_INT_t zlayer, 
            DTYPE_FLOAT_t phi[N_VERTICES], DTYPE_FLOAT_t vel[2]):
        """ Return u and v components at a point using Shepard interpolation.
        
        In FVCOM, the u and v velocity components are defined at element centres
        on sigma layers and saved at discrete points in time. Here,
        u(t,x,y,z) and v(t,x,y,z) are retrieved through i) linear interpolation
        in t and z, and ii) Shepard interpolation (which is basically a 
        special case of normalized radial basis function interpolation)
        in x and y.
        
        In Shepard interpolation, the algorithm uses velocities defined at 
        the host element's centre and its immediate neghbours (i.e. at the
        centre of those elements that share a face with the host element).
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate.
        
        ypos : float
            y-position at which to interpolate.

        zpos : float
            z-position at which to interpolate.

        host : int
            Host horizontal element.

        phi : C array, float
            Barycentric coordinates of the point.
        
        Returns:
        --------
        vel : C array, float
            Two element array giving the u and v velocity component.
        """
        # x/y coordinates of element centres
        cdef DTYPE_FLOAT_t xc[N_NEIGH_ELEMS]
        cdef DTYPE_FLOAT_t yc[N_NEIGH_ELEMS]

        # Temporary array for vel at element centres at last time point
        cdef DTYPE_FLOAT_t uc_last[N_NEIGH_ELEMS]
        cdef DTYPE_FLOAT_t vc_last[N_NEIGH_ELEMS]

        # Temporary array for vel at element centres at next time point
        cdef DTYPE_FLOAT_t uc_next[N_NEIGH_ELEMS]
        cdef DTYPE_FLOAT_t vc_next[N_NEIGH_ELEMS]

        # Vel at element centres in overlying sigma layer
        cdef DTYPE_FLOAT_t uc1[N_NEIGH_ELEMS]
        cdef DTYPE_FLOAT_t vc1[N_NEIGH_ELEMS]

        # Vel at element centres in underlying sigma layer
        cdef DTYPE_FLOAT_t uc2[N_NEIGH_ELEMS]
        cdef DTYPE_FLOAT_t vc2[N_NEIGH_ELEMS]     
        
        # Vel at the given location in the overlying sigma layer
        cdef DTYPE_FLOAT_t up1, vp1
        
        # Vel at the given location in the underlying sigma layer
        cdef DTYPE_FLOAT_t up2, vp2
        
        # Object describing a point's location within FVCOM's vertical grid. 
        cdef ZGridPosition z_grid_pos
         
        # Variables used in interpolation in time      
        cdef DTYPE_FLOAT_t time_fraction

        # Variables used in interpolation in z
        cdef DTYPE_FLOAT_t sigma_fraction, sigma_lower_layer, sigma_upper_layer

        # Array and loop indices
        cdef DTYPE_INT_t i, j, k, neighbour
        
        cdef DTYPE_INT_t nbe_min
        
        # Set variables describing the position within the vertical grid
        self._get_z_grid_position(zpos, host, zlayer, phi, &z_grid_pos)

        # Time fraction
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        nbe_min = int_min(int_min(self._nbe[0, host], self._nbe[1, host]), self._nbe[2, host])
        if nbe_min < 0:
            # Boundary element - no horizontal interpolation
            if z_grid_pos.in_vertical_boundary_layer is True:
                vel[0] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_boundary, host], self._u_next[z_grid_pos.k_boundary, host])
                vel[1] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_boundary, host], self._v_next[z_grid_pos.k_boundary, host])
                return
            else:
                up1 = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_lower_layer, host], self._u_next[z_grid_pos.k_lower_layer, host])
                vp1 = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_lower_layer, host], self._v_next[z_grid_pos.k_lower_layer, host])
                up2 = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_upper_layer, host], self._u_next[z_grid_pos.k_upper_layer, host])
                vp2 = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_upper_layer, host], self._v_next[z_grid_pos.k_upper_layer, host])
        else:
            # Non-boundary element - perform horizontal and temporal interpolation
            if z_grid_pos.in_vertical_boundary_layer is True:
                xc[0] = self._xc[host]
                yc[0] = self._yc[host]
                uc1[0] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_boundary, host], self._u_next[z_grid_pos.k_boundary, host])
                vc1[0] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_boundary, host], self._v_next[z_grid_pos.k_boundary, host])
                for i in xrange(3):
                    neighbour = self._nbe[i, host]
                    j = i+1 # +1 as host is 0
                    xc[j] = self._xc[neighbour] 
                    yc[j] = self._yc[neighbour]
                    uc1[j] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_boundary, neighbour], self._u_next[z_grid_pos.k_boundary, neighbour])
                    vc1[j] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_boundary, neighbour], self._v_next[z_grid_pos.k_boundary, neighbour])
                
                vel[0] = interp.shepard_interpolation(xpos, ypos, xc, yc, uc1)
                vel[1] = interp.shepard_interpolation(xpos, ypos, xc, yc, vc1)
                return  
            else:
                xc[0] = self._xc[host]
                yc[0] = self._yc[host]
                uc1[0] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_lower_layer, host], self._u_next[z_grid_pos.k_lower_layer, host])
                vc1[0] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_lower_layer, host], self._v_next[z_grid_pos.k_lower_layer, host])
                uc2[0] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_upper_layer, host], self._u_next[z_grid_pos.k_upper_layer, host])
                vc2[0] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_upper_layer, host], self._v_next[z_grid_pos.k_upper_layer, host])
                for i in xrange(3):
                    neighbour = self._nbe[i, host]
                    j = i+1 # +1 as host is 0
                    xc[j] = self._xc[neighbour] 
                    yc[j] = self._yc[neighbour]
                    uc1[j] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_lower_layer, host], self._u_next[z_grid_pos.k_lower_layer, host])
                    vc1[j] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_lower_layer, host], self._v_next[z_grid_pos.k_lower_layer, host])    
                    uc2[j] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_upper_layer, host], self._u_next[z_grid_pos.k_upper_layer, host])
                    vc2[j] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_upper_layer, host], self._v_next[z_grid_pos.k_upper_layer, host])
            
            # ... lower bounding sigma layer
            up1 = interp.shepard_interpolation(xpos, ypos, xc, yc, uc1)
            vp1 = interp.shepard_interpolation(xpos, ypos, xc, yc, vc1)

            # ... upper bounding sigma layer
            up2 = interp.shepard_interpolation(xpos, ypos, xc, yc, uc2)
            vp2 = interp.shepard_interpolation(xpos, ypos, xc, yc, vc2)

        # Vertical interpolation
        sigma_lower_layer = self._interp_on_sigma_layer(phi, host, z_grid_pos.k_lower_layer)
        sigma_upper_layer = self._interp_on_sigma_layer(phi, host, z_grid_pos.k_upper_layer)
        sigma_fraction = interp.get_linear_fraction_safe(zpos, sigma_lower_layer, sigma_upper_layer)

        vel[0] = interp.linear_interp(sigma_fraction, up1, up2)
        vel[1] = interp.linear_interp(sigma_fraction, vp1, vp2)
        return

    cdef _get_uv_velocity_using_linear_least_squares_interpolation(self, 
            DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, 
            DTYPE_FLOAT_t zpos, DTYPE_INT_t host, DTYPE_INT_t zlayer, 
            DTYPE_FLOAT_t phi[N_VERTICES], DTYPE_FLOAT_t vel[2]):
        """ Return u and v components at a point using LLS interpolation.
        
        In FVCOM, the u and v velocity components are defined at element centres
        on sigma layers and saved at discrete points in time. Here,
        u(t,x,y,z) and v(t,x,y,z) are retrieved through i) linear interpolation
        in t and z, and ii) Linear Least Squares (LLS) Interpolation in x and y.
        
        The LLS interpolation method uses the a1u and a2u interpolants computed
        by FVCOM (see the FVCOM manual) and saved with the model output. An
        exception to this occurs in boundary elements, where the a1u and a2u
        interpolants are set to zero. In these elements, particles "see" the
        same velocity throughout the whole element. This velocity is that which
        is defined at the element's centroid.
        
        This interpolation method can result in particles being pushed towards
        and ultimately over the land boundary.

        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate.
        
        ypos : float
            y-position at which to interpolate.

        zpos : float
            z-position at which to interpolate.

        host : int
            Host horizontal element.

        phi : C array, float
            Barycentric coordinates of the point.
        
        Returns:
        --------
        vel : C array, float
            Two element array giving the u and v velocity component.
        """
        # Temporary array for vel at element centres at last time point
        cdef DTYPE_FLOAT_t uc_last[N_NEIGH_ELEMS]
        cdef DTYPE_FLOAT_t vc_last[N_NEIGH_ELEMS]

        # Temporary array for vel at element centres at next time point
        cdef DTYPE_FLOAT_t uc_next[N_NEIGH_ELEMS]
        cdef DTYPE_FLOAT_t vc_next[N_NEIGH_ELEMS]

        # Vel at element centres in overlying sigma layer
        cdef DTYPE_FLOAT_t uc1[N_NEIGH_ELEMS]
        cdef DTYPE_FLOAT_t vc1[N_NEIGH_ELEMS]

        # Vel at element centres in underlying sigma layer
        cdef DTYPE_FLOAT_t uc2[N_NEIGH_ELEMS]
        cdef DTYPE_FLOAT_t vc2[N_NEIGH_ELEMS]     
        
        # Vel at the given location in the overlying sigma layer
        cdef DTYPE_FLOAT_t up1, vp1
        
        # Vel at the given location in the underlying sigma layer
        cdef DTYPE_FLOAT_t up2, vp2

        # Object describing a point's location within FVCOM's vertical grid. 
        cdef ZGridPosition z_grid_pos
        
        # Variables used in interpolation in time      
        cdef DTYPE_FLOAT_t time_fraction

        # Variables used in interpolation in z
        cdef DTYPE_FLOAT_t sigma_fraction, sigma_lower_layer, sigma_upper_layer

        # Array and loop indices
        cdef DTYPE_INT_t i, j, k, neighbour
        
        cdef DTYPE_INT_t nbe_min

        # Set variables describing the position within the vertical grid
        self._get_z_grid_position(zpos, host, zlayer, phi, &z_grid_pos)

        # Time fraction
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        nbe_min = int_min(int_min(self._nbe[0, host], self._nbe[1, host]), self._nbe[2, host])
        if nbe_min < 0:
            # Boundary element - no horizontal interpolation
            if z_grid_pos.in_vertical_boundary_layer is True:
                vel[0] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_boundary, host], self._u_next[z_grid_pos.k_boundary, host])
                vel[1] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_boundary, host], self._v_next[z_grid_pos.k_boundary, host])
                return
            else:
                up1 = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_lower_layer, host], self._u_next[z_grid_pos.k_lower_layer, host])
                vp1 = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_lower_layer, host], self._v_next[z_grid_pos.k_lower_layer, host])
                up2 = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_upper_layer, host], self._u_next[z_grid_pos.k_upper_layer, host])
                vp2 = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_upper_layer, host], self._v_next[z_grid_pos.k_upper_layer, host])
        else:
            # Non-boundary element - perform horizontal and temporal interpolation
            if z_grid_pos.in_vertical_boundary_layer is True:
                uc1[0] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_boundary, host], self._u_next[z_grid_pos.k_boundary, host])
                vc1[0] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_boundary, host], self._v_next[z_grid_pos.k_boundary, host])
                for i in xrange(3):
                    neighbour = self._nbe[i, host]
                    j = i+1 # +1 as host is 0
                    uc1[j] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_boundary, neighbour], self._u_next[z_grid_pos.k_boundary, neighbour])
                    vc1[j] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_boundary, neighbour], self._v_next[z_grid_pos.k_boundary, neighbour])
                
                vel[0] = self._interpolate_vel_between_elements(xpos, ypos, host, uc1)
                vel[1] = self._interpolate_vel_between_elements(xpos, ypos, host, vc1)
                return  
            else:
                uc1[0] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_lower_layer, host], self._u_next[z_grid_pos.k_lower_layer, host])
                vc1[0] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_lower_layer, host], self._v_next[z_grid_pos.k_lower_layer, host])
                uc2[0] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_upper_layer, host], self._u_next[z_grid_pos.k_upper_layer, host])
                vc2[0] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_upper_layer, host], self._v_next[z_grid_pos.k_upper_layer, host])
                for i in xrange(3):
                    neighbour = self._nbe[i, host]
                    j = i+1 # +1 as host is 0
                    uc1[j] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_lower_layer, host], self._u_next[z_grid_pos.k_lower_layer, host])
                    vc1[j] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_lower_layer, host], self._v_next[z_grid_pos.k_lower_layer, host])    
                    uc2[j] = interp.linear_interp(time_fraction, self._u_last[z_grid_pos.k_upper_layer, host], self._u_next[z_grid_pos.k_upper_layer, host])
                    vc2[j] = interp.linear_interp(time_fraction, self._v_last[z_grid_pos.k_upper_layer, host], self._v_next[z_grid_pos.k_upper_layer, host])
            
            # ... lower bounding sigma layer
            up1 = self._interpolate_vel_between_elements(xpos, ypos, host, uc1)
            vp1 = self._interpolate_vel_between_elements(xpos, ypos, host, vc1)

            # ... upper bounding sigma layer
            up2 = self._interpolate_vel_between_elements(xpos, ypos, host, uc2)
            vp2 = self._interpolate_vel_between_elements(xpos, ypos, host, vc2)
            
        # Vertical interpolation
        sigma_lower_layer = self._interp_on_sigma_layer(phi, host, z_grid_pos.k_lower_layer)
        sigma_upper_layer = self._interp_on_sigma_layer(phi, host, z_grid_pos.k_upper_layer)
        sigma_fraction = interp.get_linear_fraction_safe(zpos, sigma_lower_layer, sigma_upper_layer)

        vel[0] = interp.linear_interp(sigma_fraction, up1, up2)
        vel[1] = interp.linear_interp(sigma_fraction, vp1, vp2)
        return

    cdef DTYPE_FLOAT_t _get_omega_velocity(self, DTYPE_FLOAT_t time,
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos,
            DTYPE_INT_t host, DTYPE_INT_t zlayer, 
            DTYPE_FLOAT_t phi[N_VERTICES]):
        """ Get omega velocity through linear interpolation.
        
        Omega is defined at element nodes on sigma levels. Linear interpolation
        procedes as follows:
        
        1) Perform time interpolation of omega at nodes of the host element on
        the upper and lower bounding sigma levels.
        
        2) Interpolate omega within the host element on the upper and lower 
        bounding sigma levels.
        
        3) Perform vertical interpolation of omega between sigma levels at the
        particle's x/y position.
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate.
        
        ypos : float
            y-position at which to interpolate.

        zpos : float
            z-position at which to interpolate.

        host : int
            Host horizontal element.

        zlayer : int
            Host z layer.

        phi : C array, float
            Barycentric coordinates of the point.

        Return:
        -------
        omega : float
            Omega velocity.
        """
        # Variables used when determining indices for the sigma levels that
        # bound the particle's position
        cdef DTYPE_INT_t k_lower_level, k_upper_level
        cdef DTYPE_FLOAT_t sigma_lower_level, sigma_upper_level        
        cdef bool particle_found

        # No. of vertices and a temporary object used for determining variable
        # values at the host element's nodes
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier
        
        # Time and sigma fractions for interpolation in time and sigma        
        cdef DTYPE_FLOAT_t time_fraction, sigma_fraction
        
        # Intermediate arrays - omega
        cdef DTYPE_FLOAT_t omega_tri_t_last_lower_level[N_VERTICES]
        cdef DTYPE_FLOAT_t omega_tri_t_next_lower_level[N_VERTICES]
        cdef DTYPE_FLOAT_t omega_tri_t_last_upper_level[N_VERTICES]
        cdef DTYPE_FLOAT_t omega_tri_t_next_upper_level[N_VERTICES]
        cdef DTYPE_FLOAT_t omega_tri_lower_level[N_VERTICES]
        cdef DTYPE_FLOAT_t omega_tri_upper_level[N_VERTICES]
        
        # Intermediate arrays - zeta/h
        cdef DTYPE_FLOAT_t zeta_tri_t_last[N_VERTICES]
        cdef DTYPE_FLOAT_t zeta_tri_t_next[N_VERTICES]
        cdef DTYPE_FLOAT_t zeta_tri[N_VERTICES]
        cdef DTYPE_FLOAT_t h_tri[N_VERTICES]        
        
        # Interpolated omegas on lower and upper bounding sigma levels
        cdef DTYPE_FLOAT_t omega_lower_level
        cdef DTYPE_FLOAT_t omega_upper_level

        # Interpolated zeta/h
        cdef DTYPE_FLOAT_t zeta
        cdef DTYPE_FLOAT_t h

        # Extract omega on the lower and upper bounding sigma levels, h and zeta
        for i in xrange(N_VERTICES):
            vertex = self._nv[i,host]
            omega_tri_t_last_lower_level[i] = self._omega_last[zlayer+1, vertex]
            omega_tri_t_next_lower_level[i] = self._omega_next[zlayer+1, vertex]
            omega_tri_t_last_upper_level[i] = self._omega_last[zlayer, vertex]
            omega_tri_t_next_upper_level[i] = self._omega_next[zlayer, vertex]
            zeta_tri_t_last[i] = self._zeta_last[vertex]
            zeta_tri_t_next[i] = self._zeta_next[vertex]
            h_tri[i] = self._h[vertex]

        # Interpolate omega and zeta in time
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)
        for i in xrange(N_VERTICES):
            omega_tri_lower_level[i] = interp.linear_interp(time_fraction, 
                                                omega_tri_t_last_lower_level[i],
                                                omega_tri_t_next_lower_level[i])
            omega_tri_upper_level[i] = interp.linear_interp(time_fraction, 
                                                omega_tri_t_last_upper_level[i],
                                                omega_tri_t_next_upper_level[i])
            zeta_tri[i] = interp.linear_interp(time_fraction, zeta_tri_t_last[i], zeta_tri_t_next[i])

        # Interpolate omega, zeta and h within the host
        omega_lower_level = interp.interpolate_within_element(omega_tri_lower_level, phi)
        omega_upper_level = interp.interpolate_within_element(omega_tri_upper_level, phi)
        zeta = interp.interpolate_within_element(zeta_tri, phi)
        h = interp.interpolate_within_element(h_tri, phi)

        # Interpolate between sigma levels
        sigma_lower_level = self._interp_on_sigma_level(phi, host, zlayer+1)
        sigma_upper_level = self._interp_on_sigma_level(phi, host, zlayer)
        sigma_fraction = interp.get_linear_fraction_safe(zpos, sigma_lower_level, sigma_upper_level)

        return interp.linear_interp(sigma_fraction, omega_lower_level, omega_upper_level) / (h + zeta)

    def _read_grid(self):
        """ Set grid and coordinate variables.
        
        All communications go via the mediator in order to guarentee support for
        both serial and parallel simulations.
        
        Parameters:
        -----------
        N/A
        
        Returns:
        --------
        N/A
        """
        # Read in the grid's dimensions
        self._n_nodes = self.mediator.get_dimension_variable('node')
        self._n_elems = self.mediator.get_dimension_variable('nele')
        self._n_siglev = self.mediator.get_dimension_variable('siglev')
        self._n_siglay = self.mediator.get_dimension_variable('siglay')
        
        # Grid connectivity/adjacency
        self._nv = self.mediator.get_grid_variable('nv', (3, self._n_elems), DTYPE_INT)
        self._nbe = self.mediator.get_grid_variable('nbe', (3, self._n_elems), DTYPE_INT)

        # Cartesian coordinates
        self._x = self.mediator.get_grid_variable('x', (self._n_nodes), DTYPE_FLOAT)
        self._y = self.mediator.get_grid_variable('y', (self._n_nodes), DTYPE_FLOAT)
        self._xc = self.mediator.get_grid_variable('xc', (self._n_elems), DTYPE_FLOAT)
        self._yc = self.mediator.get_grid_variable('yc', (self._n_elems), DTYPE_FLOAT)

        # Sigma levels at nodal coordinates
        self._siglev = self.mediator.get_grid_variable('siglev', (self._n_siglev, self._n_nodes), DTYPE_FLOAT)
        
        # Sigma layers at nodal coordinates
        self._siglay = self.mediator.get_grid_variable('siglay', (self._n_siglay, self._n_nodes), DTYPE_FLOAT)

        # Bathymetry
        self._h = self.mediator.get_grid_variable('h', (self._n_nodes), DTYPE_FLOAT)

        # Interpolation parameters (a1u, a2u, aw0, awx, awy)
        self._a1u = self.mediator.get_grid_variable('a1u', (4, self._n_elems), DTYPE_FLOAT)
        self._a2u = self.mediator.get_grid_variable('a2u', (4, self._n_elems), DTYPE_FLOAT)
        
        # z min/max
        self._zmin = -1.0
        self._zmax = 0.0

    cdef _read_time_dependent_vars(self):
        """ Update time variables and memory views for FVCOM data fields.
        
        For each FVCOM time-dependent variable needed by PyLag two references
        are stored. These correspond to the last and next time points at which
        FVCOM data was saved. Together these bound PyLag's current time point.
        
        All communications go via the mediator in order to guarentee support for
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
        self._zeta_last = self.mediator.get_time_dependent_variable_at_last_time_index('zeta', (self._n_nodes), DTYPE_FLOAT)
        self._zeta_next = self.mediator.get_time_dependent_variable_at_next_time_index('zeta', (self._n_nodes), DTYPE_FLOAT)
        
        # Update memory views for u, v and w
        self._u_last = self.mediator.get_time_dependent_variable_at_last_time_index('u', (self._n_siglay, self._n_elems), DTYPE_FLOAT)
        self._u_next = self.mediator.get_time_dependent_variable_at_next_time_index('u', (self._n_siglay, self._n_elems), DTYPE_FLOAT)
        self._v_last = self.mediator.get_time_dependent_variable_at_last_time_index('v', (self._n_siglay, self._n_elems), DTYPE_FLOAT)
        self._v_next = self.mediator.get_time_dependent_variable_at_next_time_index('v', (self._n_siglay, self._n_elems), DTYPE_FLOAT)
        self._omega_last = self.mediator.get_time_dependent_variable_at_last_time_index('omega', (self._n_siglev, self._n_nodes), DTYPE_FLOAT)
        self._omega_next = self.mediator.get_time_dependent_variable_at_next_time_index('omega', (self._n_siglev, self._n_nodes), DTYPE_FLOAT)
        
        # Update memory views for kh
        if self.config.get('SIMULATION', 'vertical_random_walk_model') is not 'none':
            self._kh_last = self.mediator.get_time_dependent_variable_at_last_time_index('kh', (self._n_siglev, self._n_nodes), DTYPE_FLOAT)
            self._kh_next = self.mediator.get_time_dependent_variable_at_next_time_index('kh', (self._n_siglev, self._n_nodes), DTYPE_FLOAT)
        
        # Update memory views for viscofh
        if self.config.get('SIMULATION', 'horizontal_random_walk_model') is not 'none':
            self._viscofh_last = self.mediator.get_time_dependent_variable_at_last_time_index('viscofh', (self._n_siglay, self._n_nodes), DTYPE_FLOAT)
            self._viscofh_next = self.mediator.get_time_dependent_variable_at_next_time_index('viscofh', (self._n_siglay, self._n_nodes), DTYPE_FLOAT)
                
    cdef void _get_phi(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos,
            DTYPE_INT_t host, DTYPE_FLOAT_t phi[N_VERTICES]):
        """ Get barycentric coordinates.
        
        Parameters:
        -----------
        xpos : float
            x-position in cartesian coordinates.
        
        ypos : float
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
        interp.get_barycentric_coords(xpos, ypos, x_tri, y_tri, phi)

    cdef DTYPE_FLOAT_t _interp_on_sigma_layer(self, 
            DTYPE_FLOAT_t phi[N_VERTICES], DTYPE_INT_t host, DTYPE_INT_t kidx):
        """ Return the linearly interpolated value of sigma on the sigma layer.
        
        Compute sigma on the specified sigma layer within the given host 
        element.
        
        Parameters
        ----------
        phi : c array, float
            Array of length three giving the barycentric coordinates at which 
            to interpolate

        host : int
            Host element index

        kidx : int
            Sigma layer on which to interpolate

        Returns
        -------
        sigma: float
            Interpolated value of sigma.
        """
        cdef int vertex # Vertex identifier
        cdef DTYPE_FLOAT_t sigma_nodes[N_VERTICES]
        cdef DTYPE_FLOAT_t sigma # Sigma

        for i in xrange(N_VERTICES):
            vertex = self._nv[i,host]
            sigma_nodes[i] = self._siglay[kidx, vertex]                  

        sigma = interp.interpolate_within_element(sigma_nodes, phi)
        return sigma

    cdef DTYPE_FLOAT_t _interp_on_sigma_level(self, 
            DTYPE_FLOAT_t phi[N_VERTICES], DTYPE_INT_t host, DTYPE_INT_t kidx):
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

    cdef DTYPE_FLOAT_t _interpolate_vel_between_elements(self, 
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_INT_t host, 
            DTYPE_FLOAT_t vel_elem[N_NEIGH_ELEMS]):
        """Interpolate between elements using linear least squares interpolation.
        
        Use the a1u and a2u interpolants to compute the velocity at xpos and
        ypos.
        
        Parameters:
        -----------
        xpos : float
            x position in cartesian coordinates.

        ypos : float
            y position in cartesian coordinates.
        
        host : int
            The host element.

        vel_elem : c array, float
            Velocity at the centroid of the host element and its three 
            surrounding neighbour elements.
        """

        cdef DTYPE_FLOAT_t rx, ry
        cdef DTYPE_FLOAT_t dudx, dudy
        
        # Interpolate horizontally
        rx = xpos - self._xc[host]
        ry = ypos - self._yc[host]

        dudx = 0.0
        dudy = 0.0
        for i in xrange(N_NEIGH_ELEMS):
            dudx += vel_elem[i] * self._a1u[i, host]
            dudy += vel_elem[i] * self._a2u[i, host]
        return vel_elem[0] + dudx*rx + dudy*ry

    cdef void _get_z_grid_position(self, DTYPE_FLOAT_t zpos, DTYPE_INT_t host, 
            DTYPE_INT_t zlayer, DTYPE_FLOAT_t phi[N_VERTICES],
            ZGridPosition *z_grid_pos):
        """ Find the sigma layers bounding the given z position.
        
        First check the upper and lower boundaries, then the centre of the 
        water columnun.
        
        Parameters:
        -----------
        zpos : float
            The given z position in sigma coordinates.

        host : int
            The host element.

        zlayer : int
            The host zlayer.

        phi : c array, float
            Barycentirc coordinates within the host element.
        
        Returns:
        --------
        z_grid_pos : ZGridPostion
            Object describing the location of the given z position within 
            FVCOM's vertical grid.
        """
        cdef DTYPE_FLOAT_t sigma_test

        # Use the value of sigma on the given z layer to determine if zpos
        # lies below or above the mid-layer depth - this will determine which
        # layers should be used for interpolation.
        sigma_test = self._interp_on_sigma_layer(phi, host, zlayer)

        # Is zpos in the top or bottom boundary layer?
        if (zlayer == 0 and zpos >= sigma_test) or (zlayer == self._n_siglay - 1 and zpos <= sigma_test):
                z_grid_pos.in_vertical_boundary_layer = True
                z_grid_pos.k_boundary = zlayer

                return

        # zpos bounded by upper and lower sigma layers
        z_grid_pos.in_vertical_boundary_layer = False
        if zpos >= sigma_test:
            z_grid_pos.k_upper_layer = zlayer - 1
            z_grid_pos.k_lower_layer = zlayer
        else:
            z_grid_pos.k_upper_layer = zlayer
            z_grid_pos.k_lower_layer = zlayer + 1

        return

cdef struct ZGridPosition:
    # Struct describing the location of a point within FVCOM's vertical grid.

    # True if the given location is within the top or bottom boundary layer, 
    # False if not.
    bint in_vertical_boundary_layer

    # Top or bottom boundary layer index. Only set if the given location is
    # within the top or bottom boundary layer.    
    DTYPE_INT_t k_boundary

    # Index of the layer lying directly below the given location. Only set
    # if the given location lies within the centre of the water column.
    DTYPE_INT_t k_lower_layer

    # Index of the layer lying directly above the given location. Only set
    # if the given location lies within the centre of the water column.    
    DTYPE_INT_t k_upper_layer
