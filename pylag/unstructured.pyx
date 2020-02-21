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

# PyLag cython imports
from particle cimport Particle, to_string
from pylag.data_reader cimport DataReader
cimport pylag.interpolation as interp
from pylag.math cimport int_min, float_min, get_intersection_point
from pylag.math cimport Intersection


cdef class UnstructuredGrid:
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

    def __init__(self, config, n_nodes, n_elems, nv, nbe, x, y, xc, yc):
        self.config = config

        self.n_nodes = n_nodes
        self.n_elems = n_elems
        self.nv = nv[:]
        self.nbe = nbe[:]
        self.x = x[:]
        self.y = y[:]
        self.xc = xc[:]
        self.yc = yc[:]

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
            self.get_phi(particle.x1, particle.x2, guess, phi)

            # Check to see if the particle is in the current element
            phi_test = float_min(float_min(phi[0], phi[1]), phi[2])
            if phi_test >= 0.0:
                host_found = True

            # If the particle has walked into an element with two land
            # boundaries flag this as an error.
            if host_found is True:
                n_host_land_boundaries = 0
                for i in xrange(3):
                    if self.nbe[i,guess] == -1:
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
                guess = self.nbe[0,last_guess]
            elif phi[1] == phi_test:
                guess = self.nbe[1,last_guess]
            else:
                guess = self.nbe[2,last_guess]

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
                vertex = self.nv[i,elem]
                x_tri[i] = self.x[vertex]
                y_tri[i] = self.y[vertex]

            # This keeps track of the element currently being checked
            current_elem = elem

            # Loop over all sides of the element to find the land boundary the element crossed
            for i in xrange(3):
                x1_idx = x1_indices[i]
                x2_idx = x2_indices[i]
                nbe_idx = nbe_indices[i]

                # Test to avoid checking the side the pathline just crossed
                if last_elem == self.nbe[nbe_idx, elem]:
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
                elem = self.nbe[nbe_idx, elem]

                # Check to see if the pathline has exited the domain
                if elem >= 0:
                    # Treat elements with two boundaries as land (i.e. set
                    # `flag' equal to -1) and return the last element checked
                    n_host_boundaries = 0
                    for i in xrange(3):
                        if self.nbe[i,elem] == -1:
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

        for guess in xrange(self.n_elems):
            # Barycentric coordinates
            self.get_phi(particle.x1, particle.x2, guess, phi)

            # Check to see if the particle is in the current element
            phi_test = float_min(float_min(phi[0], phi[1]), phi[2])
            if phi_test >= 0.0:
                host_found = True

            if host_found is True:
                # If the element has two land boundaries, flag the particle as
                # being outside of the domain
                n_host_land_boundaries = 0
                for i in xrange(3):
                    if self.nbe[i,guess] == -1:
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
            vertex = self.nv[i, particle_new.host_horizontal_elem]
            x_tri[i] = self.x[vertex]
            y_tri[i] = self.y[vertex]

        # Loop over all sides of the element to find the land boundary the element crossed
        for i in xrange(3):
            x1_idx = x1_indices[i]
            x2_idx = x2_indices[i]
            nbe_idx = nbe_indices[i]

            nbe = self.nbe[nbe_idx, particle_new.host_horizontal_elem]

            if nbe != -1:
                # Compute the number of land boundaries the neighbour has - elements with two
                # land boundaries are themselves treated as land
                n_land_boundaries = 0
                for i in xrange(3):
                    if self.nbe[i, nbe] == -1:
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
        particle.x1 = self.xc[particle.host_horizontal_elem]
        particle.x2 = self.yc[particle.host_horizontal_elem]
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

        self.get_phi(particle.x1, particle.x2,
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

    cdef void get_phi(self, DTYPE_FLOAT_t x1, DTYPE_FLOAT_t x2,
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
            vertex = self.nv[i,host]
            x_tri[i] = self.x[vertex]
            y_tri[i] = self.y[vertex]

        # Calculate barycentric coordinates
        interp.get_barycentric_coords(x1, x2, x_tri, y_tri, phi)

    cdef void get_grad_phi(self, DTYPE_INT_t host,
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
            vertex = self.nv[i,host]
            x_tri[i] = self.x[vertex]
            y_tri[i] = self.y[vertex]

        # Calculate gradient in barycentric coordinates
        interp.get_barycentric_gradients(x_tri, y_tri, dphi_dx, dphi_dy)
