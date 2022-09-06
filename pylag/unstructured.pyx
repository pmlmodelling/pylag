"""
Tools for navigating unstructured triangular grids

Note
----
unstructured is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""


include "constants.pxi"

from libcpp.vector cimport vector
from libc.math cimport fabs

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

# PyLag python imports
from pylag.particle_cpp_wrapper cimport ParticleSmartPtr
from pylag.math import geographic_to_cartesian_coords_python
from pylag.exceptions import PyLagRuntimeError

# PyLag cython imports
from pylag.parameters cimport deg_to_radians, radians_to_deg, earth_radius, pi
from pylag.particle cimport Particle
from pylag.particle_cpp_wrapper cimport to_string
from pylag.data_reader cimport DataReader
cimport pylag.interpolation as interp
from pylag.math cimport geographic_to_cartesian_coords, rotate_axes, det_third_order
from pylag.math cimport great_circle_arc_segments_intersect
from pylag.math cimport haversine
from pylag.math cimport int_min, float_min, get_intersection_point, get_intersection_point_in_geographic_coordinates
from pylag.math cimport area_of_a_triangle, area_of_a_spherical_triangle

cdef class Grid:
    """ Base class for grid objects

    Objects of type Grid can perform grid searches, compute local coordinates to assist
    with interpolation and help identify grid boundary crossings. Derived classes must
    implement functionality that is specific to a given grid type (e.g. an unstructured
    grid in cartesian coordinates vs an unstructured grid in geographic coordinates.
    """
    def find_host_using_global_search_wrapper(self,
                                              ParticleSmartPtr particle):
        """ Python wrapper for finding and setting the host element using a global search

        Parameters
        ----------
        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : int
             Flag signifying whether the host element was found successfully.
        """
        return self.find_host_using_global_search(particle.get_ptr())

    cdef DTYPE_INT_t find_host_using_global_search(self,
                                                   Particle *particle) except INT_ERR:
        raise NotImplementedError

    def find_host_using_local_search_wrapper(self,
                                             ParticleSmartPtr particle):
        """ Python wrapper for finding and setting the host element using a local search

        Parameters
        ----------
        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : int
             Flag signifying whether the host element was found successfully.
        """
        return self.find_host_using_local_search(particle.get_ptr())

    cdef DTYPE_INT_t find_host_using_local_search(self,
                                                  Particle *particle) except INT_ERR:
        raise NotImplementedError

    def find_host_using_particle_tracing_wrapper(self, ParticleSmartPtr particle_old,
                                  ParticleSmartPtr particle_new):
        """ Python wrapper for finding and setting the host element using particle tracing

        Parameters
        ----------
        particle_old : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its last known position.

        particle_new : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its new position.

        Returns
        -------
         : int
             Flag signifying whether the host element was found successfully.
        """
        return self.find_host_using_particle_tracing(particle_old.get_ptr(), particle_new.get_ptr())

    cdef DTYPE_INT_t find_host_using_particle_tracing(self, Particle *particle_old,
                                                      Particle *particle_new) except INT_ERR:
        raise NotImplementedError

    def get_boundary_intersection_wrapper(self, ParticleSmartPtr particle_old,
                                  ParticleSmartPtr particle_new):
        """ Python wrapper for finding the point no the boundary the particle intersected

        Parameters
        ----------
        particle_old : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its last known position.

        particle_new : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its new position.

        Returns
        -------
        start_point : NumPy array
            Start coordinates of the side the particle crossed.

        end_point : NumPy array
            End coordinates of the side the particle crossed.

        intersection : NumPy array
            Coordinates of the intersection point.

        """
        cdef DTYPE_FLOAT_t start_point_c[2]
        cdef DTYPE_FLOAT_t end_point_c[2]
        cdef DTYPE_FLOAT_t intersection_c[2]

        self.get_boundary_intersection(particle_old.get_ptr(), particle_new.get_ptr(), start_point_c,
                                       end_point_c, intersection_c)

        start_point, end_point, intersection = [], [], []
        for i in range(2):
            start_point.append(start_point_c[i])
            end_point.append(end_point_c[i])
            intersection.append(intersection_c[i])

        return np.array(start_point), np.array(end_point), np.array(intersection)

    cdef get_boundary_intersection(self,
                                   Particle *particle_old,
                                   Particle *particle_new,
                                   DTYPE_FLOAT_t start_point[2],
                                   DTYPE_FLOAT_t end_point[2],
                                   DTYPE_FLOAT_t intersection[2]):
        raise NotImplementedError

    def set_default_location_wrapper(self, ParticleSmartPtr particle):
        """ Python wrapper for setting the default location of a particle

        Can be used to set a particle's position to a default location
        within an element (for example, should it not be possible to
        strictly apply the horizontal boundary condition).

        Parameters
        ----------
        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : None
        """
        return self.set_default_location(particle.get_ptr())

    cdef set_default_location(self, Particle *particle):
        raise NotImplementedError

    def set_local_coordinates_wrapper(self, ParticleSmartPtr particle):
        """ Python wrapper for setting particle local coordinates

        Used to set particle local horizontal coordinates (e.g. within its host
        element).

        Parameters
        ----------
        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : None
        """
        return self.set_local_coordinates(particle.get_ptr())

    cdef set_local_coordinates(self, Particle *particle):
        raise NotImplementedError

    def get_element_area_wrapper(self, ParticleSmartPtr particle):
        """ Python wrapper for returning the area of the element in which the particle resides.

        Parameters
        ----------
        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : float
            The element area.
        """
        return self.get_element_area(particle.get_ptr())

    cdef DTYPE_FLOAT_t get_element_area(self, Particle *particle) except FLOAT_ERR:
        raise NotImplementedError

    cpdef vector[DTYPE_FLOAT_t] get_phi(self, const DTYPE_FLOAT_t &x1, const DTYPE_FLOAT_t &x2, const DTYPE_INT_t &host):
        raise NotImplementedError

    def get_grad_phi_wrapper(self, host):
        """ Python wrapper for computing gradients in phi

        Parameters
        ----------
        host : int
            The host element.

        Returns
        -------
        dphi_dx, dphi_dy : NDArray
             Phi gradients in x and y.
        """
        cdef DTYPE_FLOAT_t dphi_dx_c[3]
        cdef DTYPE_FLOAT_t dphi_dy_c[3]
        cdef DTYPE_INT_t i

        self.get_grad_phi(host, dphi_dx_c, dphi_dy_c)

        dphi_dx, dphi_dy = [], []
        for i in range(3):
            dphi_dx.append(dphi_dx_c[i])
            dphi_dy.append(dphi_dy_c[i])

        return np.array(dphi_dx), np.array(dphi_dy)

    cdef void get_grad_phi(self, DTYPE_INT_t host,
                           DTYPE_FLOAT_t dphi_dx[3],
                           DTYPE_FLOAT_t dphi_dy[3]) except *:
        raise NotImplementedError

    def interpolate_in_space_wrapper(self, var_arr, ParticleSmartPtr particle):
        """ Python wrapper for interpolate in space

        Parameters
        ----------
        var_arr : numpy array
            Variable array of points defined at element nodes.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.
        """
        return self.interpolate_in_space(var_arr, particle.get_ptr())

    cdef DTYPE_FLOAT_t interpolate_in_space(self, DTYPE_FLOAT_t[::1] var_arr, Particle *particle) except FLOAT_ERR:
        raise NotImplementedError

    def interpolate_in_time_and_space_2D_wrapper(self, var_last_arr, var_next_arr, time_fraction,
                                                 ParticleSmartPtr particle):
        """ Python wrapper for interpolate in time and space 2D

        Parameters
        ----------
        var_last_arr : numpy array
            Variable array of points defined at element nodes at the last time index.

        var_next_arr : numpy array
            Variable array of points defined at element nodes at the next time index.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.
        """
        return self.interpolate_in_time_and_space_2D(var_last_arr, var_next_arr, time_fraction, particle.get_ptr())

    cdef DTYPE_FLOAT_t interpolate_in_time_and_space_2D(self, DTYPE_FLOAT_t[::1] var_last_arr,
                                                        DTYPE_FLOAT_t[::1] var_next_arr,
                                                        DTYPE_FLOAT_t time_fraction, Particle *particle) except FLOAT_ERR:
        raise NotImplementedError

    def interpolate_in_time_and_space_wrapper(self, var_last_arr, var_next_arr, k, time_fraction, ParticleSmartPtr particle):
        """ Python wrapper for interpolate in time and space

        Parameters
        ----------
        var_last_arr : numpy array
            Variable array of points defined at element nodes at the last time point.

        var_last_arr : numpy array
            Variable array of points defined at element nodes at the next time point.

        k : int
            k layer index.

        time_fraction : float
            Fraction position between the first and last time points.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.
        """
        return self.interpolate_in_time_and_space(var_last_arr, var_next_arr, k, time_fraction, particle.get_ptr())

    cdef DTYPE_FLOAT_t interpolate_in_time_and_space(self, DTYPE_FLOAT_t[:, ::1] var_last_arr,
                                                     DTYPE_FLOAT_t[:, ::1] var_next_arr, DTYPE_INT_t k,
                                                     DTYPE_FLOAT_t time_fraction, Particle *particle) except FLOAT_ERR:
        raise NotImplementedError

    cdef void interpolate_grad_in_time_and_space(self, const DTYPE_FLOAT_t[:, ::1] &var_last_arr,
                                                 const DTYPE_FLOAT_t[:, ::1] &var_next_arr, DTYPE_INT_t k,
                                                 DTYPE_FLOAT_t time_fraction, Particle *particle,
                                                 DTYPE_FLOAT_t var_prime[2]) except *:
        raise NotImplementedError

    def shepard_interpolation_wrapper(self, const DTYPE_FLOAT_t &x,
            const DTYPE_FLOAT_t &y, const vector[DTYPE_FLOAT_t] &xpts,
            const vector[DTYPE_FLOAT_t] &ypts, const vector[DTYPE_FLOAT_t] &vals,
            const vector[DTYPE_INT_t] &valid_points):
        """ Python wrapper for shepard interpolation

        Parameters
        ----------
        x : float
            x-position

        y - float
            y-position

        xpts : numpy array
            x-coordinates of values.

        ypts : numpy array
            y-coordinates of values.

        vals : numpy array
            Values at x/y coordinates.

        valid_points : numpy array
            Array of 1/0 flags which signify which values to use (1) and which to ignore (0).
        """
        cdef DTYPE_FLOAT_t xpts_c[4], ypts_c[4], vals_c[4]
        cdef DTYPE_INT_t valid_points_c[4]
        cdef int i

        if xpts.size() != 4 or ypts.size() != 4 or vals.size() != 4 or valid_points.size() != 4:
            raise ValueError('Input arrays should be of length 4.')

        for i in range(4):
            xpts_c[i] = xpts[i]
            ypts_c[i] = ypts[i]
            vals_c[i] = vals[i]
            valid_points_c[i] = valid_points[i]

        return self.shepard_interpolation(x, y, xpts_c, ypts_c, vals_c, valid_points_c)

    cdef DTYPE_FLOAT_t shepard_interpolation(self, const DTYPE_FLOAT_t &x,
            const DTYPE_FLOAT_t &y, const DTYPE_FLOAT_t xpts[4], const DTYPE_FLOAT_t ypts[4],
            const DTYPE_FLOAT_t vals[4], const DTYPE_INT_t valid_points[4]) except FLOAT_ERR:
        raise NotImplementedError


cdef class UnstructuredCartesianGrid(Grid):
    """ Unstructured grid

    Objects of type UnstructuredCartesianGrid can perform grid searches,
    compute local coordinates to assist with interpolation and help identify
    grid boundary crossings for unstructured cartesian (x,y) grids.

    Parameters
    ----------
    config : configparser.ConfigParser
        PyLag configuration object.

    name : str
        The grid name. Useful if data are defined on multiple unstructured grids
        and a means is required to distinguish one from the other.

    n_nodes : int
        The number of nodes

    n_elems : int
        The number of elements

    nv : memoryview
        Memory view of nodes surrounding elements. With shape [3, n_elems]

    nbe : memoryview
        Memory view of elements surrounding elements. With shape [3, n_elems]

    x : 1D memory view
        x-coordinates of grid nodes

    y : 1D memory view
        y-coordinates of grid nodes

    xc : 1D memory view
        x-coordinates of element centres

    yc : 1D memory view
        y-coordinates of element centres

    land_sea_mask_c : 1D memory view
        Land-sea element mask

    land_sea_mask : 1D memory view
        Land-sea element mask nodes
    """
    # Configurtion object
    cdef object config

    # The grid name
    cdef string name

    # Grid dimensions
    cdef DTYPE_INT_t n_elems, n_nodes

    # Element connectivity
    cdef DTYPE_INT_t[:,::1] nv

    # Element adjacency
    cdef DTYPE_INT_t[:,::1] nbe

    # Nodal coordinates
    cdef DTYPE_FLOAT_t[::1] x
    cdef DTYPE_FLOAT_t[::1] y

    # Element centre coordinates
    cdef DTYPE_FLOAT_t[::1] xc
    cdef DTYPE_FLOAT_t[::1] yc

    # Land sea mask
    cdef DTYPE_INT_t[::1] land_sea_mask_c
    cdef DTYPE_INT_t[::1] land_sea_mask

    # Barycentric gradients
    cdef DTYPE_INT_t[:] barycentric_gradients_have_been_cached
    cdef DTYPE_FLOAT_t[:, ::1] dphi_dx
    cdef DTYPE_FLOAT_t[:, ::1] dphi_dy

    # Cell areas
    cdef DTYPE_FLOAT_t[::1] areas

    def __init__(self, config, name, n_nodes, n_elems, nv, nbe, x, y, xc, yc, land_sea_mask_c, land_sea_mask, areas=None):
        self.config = config

        self.name = name
        self.n_nodes = n_nodes
        self.n_elems = n_elems
        self.nv = nv
        self.nbe = nbe
        self.x = x
        self.y = y
        self.xc = xc
        self.yc = yc
        self.land_sea_mask_c = land_sea_mask_c
        self.land_sea_mask = land_sea_mask
        if areas is not None:
            self.areas = areas

        # Containers for preserving the value of gradient calculations
        self.barycentric_gradients_have_been_cached = np.zeros(self.n_elems, dtype=DTYPE_INT, order='C')
        self.dphi_dx = np.ones((self.n_elems, 3), dtype=DTYPE_FLOAT, order='C') * -999.
        self.dphi_dy = np.ones((self.n_elems, 3), dtype=DTYPE_FLOAT, order='C') * -999.

    cdef DTYPE_INT_t find_host_using_local_search(self, Particle *particle) except INT_ERR:
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

        Parameters
        ----------
        particle: *Particle
            The particle.

        Returns
        -------
        flag : int
            Integer flag that indicates whether or not the seach was successful.
        """
        # Intermediate arrays/variables
        cdef vector[DTYPE_FLOAT_t] phi
        cdef DTYPE_FLOAT_t phi_test

        cdef bint host_found

        cdef DTYPE_INT_t n_host_land_boundaries

        cdef DTYPE_INT_t flag, guess, last_guess, second_to_last_guess

        # Check for non-sensical start points.
        guess = particle.get_host_horizontal_elem(self.name)
        if guess < 0:
            raise ValueError('Invalid start point for local host element '\
                    'search. Start point = {}'.format(guess))

        host_found = False
        last_guess = -1
        second_to_last_guess = -1

        while True:
            # Barycentric coordinates
            phi = self.get_phi(particle.get_x1(), particle.get_x2(), guess)

            # Check to see if the particle is in the current element
            phi_test = float_min(float_min(phi[0], phi[1]), phi[2])
            if phi_test >= 0.0:
                host_found = True

            # Check to see if the particle has walked into an invalid element (e.g. an
            # element treated as land)
            if host_found is True:
                if self.land_sea_mask_c[guess] != LAND:
                    # Normal element
                    particle.set_host_horizontal_elem(self.name, guess)

                    particle.set_phi(self.name, phi)

                    return IN_DOMAIN
                else:
                    # Element is masked
                    return IN_MASKED_ELEM

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
        cdef int i # Loop counter
        cdef int vertex # Vertex identifier
        cdef DTYPE_INT_t elem, last_elem, current_elem # Element identifies
        cdef DTYPE_INT_t flag, host

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

        # Initialise arrays
        x1_indices[:] = [0,1,2]
        x2_indices[:] = [1,2,0]
        nbe_indices[:] = [2,0,1]

        # Construct arrays to hold the coordinates of the particle's previous
        # position vector and its new position vector
        x3[0] = particle_old.get_x1(); x3[1] = particle_old.get_x2()
        x4[0] = particle_new.get_x1(); x4[1] = particle_new.get_x2()

        # Start the search using the host known to contain (x1_old, x2_old)
        elem = particle_old.get_host_horizontal_elem(self.name)

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

                if get_intersection_point(x1, x2, x3, x4, xi) == 0:
                    # Lines do not intersect - check the next one
                    continue

                # Intersection found - keep a record of the last element checked
                last_elem = elem

                # Index for the neighbour element
                elem = self.nbe[nbe_idx, elem]

                # Check to see if the pathline has exited the domain
                if elem >= 0:
                    if self.land_sea_mask_c[elem] == LAND:
                        flag = LAND_BDY_CROSSED

                        # Set host to the last element the particle passed through
                        particle_new.set_host_horizontal_elem(self.name, last_elem)
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
                    particle_new.set_host_horizontal_elem(self.name, last_elem)

                    return flag

            if current_elem == elem:
                # Particle has not exited the current element meaning it must
                # still reside in the domain
                flag = IN_DOMAIN
                particle_new.set_host_horizontal_elem(self.name, current_elem)
                self.set_local_coordinates(particle_new)

                return flag

    cdef DTYPE_INT_t find_host_using_global_search(self, Particle *particle) except INT_ERR:
        """ Returns the host horizontal element through global searching.

        Sequentially search all elements for the given location. Set the particle
        host element if found.

        Parameters
        ----------
        particle_old: *Particle
            The particle.

        Returns
        -------
        flag : int
            Integer flag that indicates whether or not the seach was successful.
        """
        # Intermediate arrays/variables
        cdef vector[DTYPE_FLOAT_t] phi
        cdef DTYPE_FLOAT_t phi_test

        cdef bint host_found

        cdef DTYPE_INT_t n_host_land_boundaries

        cdef DTYPE_INT_t i, guess

        host_found = False

        for guess in xrange(self.n_elems):
            # Barycentric coordinates
            phi = self.get_phi(particle.get_x1(), particle.get_x2(), guess)

            # Check to see if the particle is in the current element
            phi_test = float_min(float_min(phi[0], phi[1]), phi[2])
            if phi_test >= 0.0:
                host_found = True

            if host_found is True:
                if self.land_sea_mask_c[guess] != LAND:
                    particle.set_host_horizontal_elem(self.name, guess)

                    particle.set_phi(self.name, phi)

                    return IN_DOMAIN
                else:
                    return BDY_ERROR
        return BDY_ERROR

    cdef get_boundary_intersection(self,
                                   Particle *particle_old,
                                   Particle *particle_new,
                                   DTYPE_FLOAT_t start_point[2],
                                   DTYPE_FLOAT_t end_point[2],
                                   DTYPE_FLOAT_t intersection[2]):
        """ Find the boundary intersection point

        This function is primarily intended to assist in the application of
        horizontal boundary conditions where it is often necessary to establish
        the point on a side of an element at which particle crossed before
        exiting the model domain.

        Parameters
        ----------
        particle_old: *Particle
            The particle at its old position.

        particle_new: *Particle
            The particle at its new position.

        start_point : vector[float]
            Start coordinates of the side the particle crossed.

        end_point : vector[float]
            End coordinates of the side the particle crossed.

        intersection : vector[float]
            Coordinates of the intersection point.

        Returns
        -------
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

        # Initialise arrays
        x1_indices[:] = [0,1,2]
        x2_indices[:] = [1,2,0]
        nbe_indices[:] = [2,0,1]

        # Construct arrays to hold the coordinates of the particle's previous
        # position vector and its new position vector
        x3[0] = particle_old.get_x1(); x3[1] = particle_old.get_x2()
        x4[0] = particle_new.get_x1(); x4[1] = particle_new.get_x2()

        # Extract nodal coordinates
        for i in xrange(3):
            vertex = self.nv[i, particle_new.get_host_horizontal_elem(self.name)]
            x_tri[i] = self.x[vertex]
            y_tri[i] = self.y[vertex]

        # Loop over all sides of the element to find the land boundary the element crossed
        for i in xrange(3):
            x1_idx = x1_indices[i]
            x2_idx = x2_indices[i]
            nbe_idx = nbe_indices[i]

            nbe = self.nbe[nbe_idx, particle_new.get_host_horizontal_elem(self.name)]

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

            if get_intersection_point(x1, x2, x3, x4, xi) == 1:
                for i in range(2):
                    start_point[i] = x1[i]
                    end_point[i] = x2[i]
                    intersection[i] = xi[i]
                return
            else:
                continue

        raise RuntimeError('Failed to calculate boundary intersection.')

    cdef set_default_location(self, Particle *particle):
        """ Set default location

        Move the particle to its host element's centroid.
        """
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self.name)

        particle.set_x1(self.xc[host_element])
        particle.set_x2(self.yc[host_element])
        self.set_local_coordinates(particle)
        return

    cdef set_local_coordinates(self, Particle *particle):
        """ Set local coordinates

        Each particle has associated with it a set of global coordinates
        and a set of local coordinates. Here, the global coordinates and the
        host horizontal element are used to set the local coordinates.

        Parameters
        ----------
        particle: *Particle
            Pointer to a Particle struct
        """
        cdef vector[DTYPE_FLOAT_t] phi

        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self.name)

        cdef DTYPE_INT_t i

        phi = self.get_phi(particle.get_x1(), particle.get_x2(), host_element)

        # Check for negative values.
        for i in xrange(3):
            if phi[i] >= 0.0:
                continue
            elif phi[i] >= -EPSILON:
                phi[i] = 0.0
            else:
                s = to_string(particle)
                msg = "One or more local coordinates are invalid (phi = {}) \n\n"\
                      "The following information may be used to study the \n"\
                      "failure in more detail. \n\n"\
                      "{}".format(phi[i], s)
                print(msg)

                raise ValueError('One or more local coordinates are negative')

        # Set phi
        particle.set_phi(self.name, phi)

    cdef DTYPE_FLOAT_t get_element_area(self, Particle *particle) except FLOAT_ERR:
        """ Return the element's area within which the particle resides

        Parameters
        ----------
        particle: *Particle
            Pointer to a Particle object
        """
        # Vertex identifies
        cdef int vertex_1, vertex_2, vertex_3

        # Nodal coordinates [x, y]
        cdef DTYPE_FLOAT_t x1[2]
        cdef DTYPE_FLOAT_t x2[2]
        cdef DTYPE_FLOAT_t x3[2]

        cdef DTYPE_INT_t host

        cdef DTYPE_FLOAT_t area

        host = particle.get_host_horizontal_elem(self.name)

        if self.areas is not None:
            area = self.areas[host]
        else:
            # Extract nodal cartesian coordinates
            vertex_1 = self.nv[0, host]
            vertex_2 = self.nv[1, host]
            vertex_3 = self.nv[2, host]
            x1[0] = self.x[vertex_1]
            x1[1] = self.y[vertex_1]
            x2[0] = self.x[vertex_2]
            x2[1] = self.y[vertex_2]
            x3[0] = self.x[vertex_3]
            x3[1] = self.y[vertex_3]

            area = area_of_a_triangle(x1, x2, x3)

        return area

    cpdef vector[DTYPE_FLOAT_t] get_phi(self, const DTYPE_FLOAT_t &x1, const DTYPE_FLOAT_t &x2, const DTYPE_INT_t &host):
        """ Get barycentric coordinates.

        Compute and return barycentric coordinates for the point (x,y) within the
        2D triangle defined by x/y coordinates stored in the vectors x_tri and y_tri.

        Barycentric coordinates are calculated using the formula:

        phi_i(x,y) = A_i(x,y)/A

        where A is the area of the element and A_i is the area of the sub triangle
        that is formed using the particle's position coordinates within the element
        and the two element vertices that lie opposite the `i` vertex. Signed areas
        are calculated using the vector cross product:

        2A(abc) = (x1 - x0)(y2 - y0) - (y1 - y0)(x2 - x0)

        A_i terms are computed by substituting in x and y for x_i and y_i for i=0:2.

        In PyLag, nodes are clockwise ordered, meaning the signed area given by the
        above formula is negative. However the negative sign and the factor of two
        cancel when the ratio is formed. For this reason, they are both ignored.

        Parameters
        ----------
        x1 : float
            x-position in cartesian coordinates.

        x2 : float
            y-position in cartesian coordinates.

        host : int
            Host element.

        Returns
        -------
        phi : C array, float
            Barycentric coordinates.
        """
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier

        # Intermediate arrays
        cdef DTYPE_FLOAT_t x_tri[3]
        cdef DTYPE_FLOAT_t y_tri[3]

        cdef vector[DTYPE_FLOAT_t] phi = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)

        cdef DTYPE_FLOAT_t a1, a2, a3, a4, twice_signed_element_area

        for i in xrange(N_VERTICES):
            vertex = self.nv[i,host]
            x_tri[i] = self.x[vertex]
            y_tri[i] = self.y[vertex]

        # Intermediate terms
        a1 = x_tri[1] - x_tri[0]
        a2 = y_tri[2] - y_tri[0]
        a3 = y_tri[1] - y_tri[0]
        a4 = x_tri[2] - x_tri[0]

        # Evaluate the vector cross product
        twice_signed_element_area = a1 * a2 - a3 * a4

        # Transformation to barycentric coordinates
        phi[2] = (a1*(x2 - y_tri[0]) - a3*(x1 - x_tri[0]))/twice_signed_element_area
        phi[1] = (a2*(x1 - x_tri[2]) - a4*(x2 - y_tri[2]))/twice_signed_element_area
        phi[0] = 1.0 - phi[1] - phi[2]

        return phi

    cdef void get_grad_phi(self, DTYPE_INT_t host,
                           DTYPE_FLOAT_t dphi_dx[3],
                           DTYPE_FLOAT_t dphi_dy[3]) except *:
        """ Get gradient in phi with respect to x and y

        Gradients in phi are calculated as described in Lynch (2015), p. 232. As
        phi is linear within the element, the gradient is a constant. The terms
        are calculated using the formulas:

        dphi_i/dx = Delta(y_i) / 2A

        dphi_i/dy = -Delta(x_i) / 2A

        where A is the element area and the Delta terms are given by:

        Delta(x1) = x2 - x3, etc
        Delta(y1) = y2 - y3, etc

        Note the formulas presented in Lynch assume an anticlockwise ordering of
        elements whereas PyLag adopts clockwise ordering. However, as the two
        terms are computed using ratios, the minus signs cancel making it
        possible to use the same formulas.

        Parameters
        ----------
        host : int
            Host element

        dphi_dx : C array, float
            Gradient with respect to x

        dphi_dy : C array, float
            Gradient with respect to y
        """

        cdef DTYPE_INT_t i # Loop counters
        cdef DTYPE_INT_t vertex # Vertex identifier

        # Intermediate arrays
        cdef DTYPE_FLOAT_t x_tri[3]
        cdef DTYPE_FLOAT_t y_tri[3]

        cdef DTYPE_FLOAT_t a1, a2, a3, a4, twice_signed_element_area

        # Return cached values if they have been pre-computed
        if self.barycentric_gradients_have_been_cached[host] == 1:
            for i in range(3):
                dphi_dx[i] = self.dphi_dx[host, i]
                dphi_dy[i] = self.dphi_dy[host, i]
            return

        for i in xrange(3):
            vertex = self.nv[i,host]
            x_tri[i] = self.x[vertex]
            y_tri[i] = self.y[vertex]

        # Intermediate terms
        a1 = x_tri[1] - x_tri[0]
        a2 = y_tri[2] - y_tri[0]
        a3 = y_tri[1] - y_tri[0]
        a4 = x_tri[2] - x_tri[0]

        # Evaluate the vector cross product
        twice_signed_element_area = a1 * a2 - a3 * a4

        dphi_dx[0] = (y_tri[1] - y_tri[2])/twice_signed_element_area
        dphi_dx[1] = (y_tri[2] - y_tri[0])/twice_signed_element_area
        dphi_dx[2] = (y_tri[0] - y_tri[1])/twice_signed_element_area

        dphi_dy[0] = (x_tri[2] - x_tri[1])/twice_signed_element_area
        dphi_dy[1] = (x_tri[0] - x_tri[2])/twice_signed_element_area
        dphi_dy[2] = (x_tri[1] - x_tri[0])/twice_signed_element_area

        # Cache values
        self.barycentric_gradients_have_been_cached[host] = 1
        for i in range(3):
            self.dphi_dx[host, i] = dphi_dx[i]
            self.dphi_dy[host, i] = dphi_dy[i]

    cdef DTYPE_FLOAT_t interpolate_in_space(self, DTYPE_FLOAT_t[::1] var_arr,  Particle* particle) except FLOAT_ERR:
        """ Interpolate the given field in space

        Interpolate the given field on the horizontal grid. The supplied fields
        should be 1D arrays of values defined at element nodes.

        Parameters
        ----------
        var_last_arr : 1D MemoryView
            Array of variable values at the last time index.

        var_next_arr : 1D MemoryView
            Array of variable values at the next time index.

        particle: *Particle
            Pointer to a Particle object.

        Returns
        -------
         : float
            The interpolated value of the variable
        """
        cdef DTYPE_INT_t vertex # Vertex identifier
        cdef DTYPE_FLOAT_t var_nodes[3]
        cdef DTYPE_FLOAT_t phi[3]

        # Phi
        cdef const vector[DTYPE_FLOAT_t] *_phi = &particle.get_phi(self.name)

        # Host element
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self.name)

        cdef int i

        for i in xrange(N_VERTICES):
            vertex = self.nv[i, host_element]
            var_nodes[i] = var_arr[vertex]
            phi[i] = _phi.at(i)

        return interp.interpolate_within_element(var_nodes, phi)

    cdef DTYPE_FLOAT_t interpolate_in_time_and_space_2D(self, DTYPE_FLOAT_t[::1] var_last_arr,
                                                        DTYPE_FLOAT_t[::1] var_next_arr,
                                                        DTYPE_FLOAT_t time_fraction, Particle* particle) except FLOAT_ERR:
        """ Interpolate the given field in time and space 2D

        Interpolate the given field in time and space on the horizontal grid. The supplied fields
        should be 1D arrays of values defined at element nodes.

        Parameters
        ----------
        var_last_arr : 1D MemoryView
            Array of variable values at the last time index.

        var_next_arr : 1D MemoryView
            Array of variable values at the next time index.

        time_fraction : float
            Time interpolation coefficient

        particle: *Particle
            Pointer to a Particle object.

        Returns
        -------
         : float
            The interpolated value of the variable
        """
        cdef DTYPE_INT_t vertex # Vertex identifier
        cdef DTYPE_FLOAT_t var_last, var_next
        cdef DTYPE_FLOAT_t var_nodes[3]
        cdef DTYPE_FLOAT_t phi[3]

        # Phi
        cdef const vector[DTYPE_FLOAT_t] *_phi = &particle.get_phi(self.name)

        # Host element
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self.name)

        cdef DTYPE_INT_t i

        for i in xrange(N_VERTICES):
            vertex = self.nv[i, host_element]
            var_last = var_last_arr[vertex]
            var_next = var_next_arr[vertex]

            if var_last != var_next:
                var_nodes[i] = interp.linear_interp(time_fraction, var_last, var_next)
            else:
                var_nodes[i] = var_last

            phi[i] = _phi.at(i)

        return interp.interpolate_within_element(var_nodes, phi)

    cdef DTYPE_FLOAT_t interpolate_in_time_and_space(self, DTYPE_FLOAT_t[:, ::1] var_last_arr,
                                                     DTYPE_FLOAT_t[:, ::1] var_next_arr, DTYPE_INT_t k,
                                                     DTYPE_FLOAT_t time_fraction, Particle* particle) except FLOAT_ERR:
        """ Interpolate the given field in time and space

        Interpolate the given field in time and space on the horizontal grid. The supplied fields
        should be 2D arrays of values defined at element nodes.

        Parameters
        ----------
        var_last_arr : 2D MemoryView
            Array of variable values at the last time index.

        var_next_arr : 2D MemoryView
            Array of variable values at the next time index.

        k : int
            k layer index.

        time_fraction : float
            Time interpolation coefficient

        particle: *Particle
            Pointer to a Particle object.

        Returns
        -------
         : float
            The interpolated value of the variable
        """
        cdef DTYPE_INT_t vertex # Vertex identifier
        cdef DTYPE_FLOAT_t var_last, var_next
        cdef DTYPE_FLOAT_t var_nodes[3]
        cdef DTYPE_FLOAT_t phi[3]

        # Phi
        cdef const vector[DTYPE_FLOAT_t] *_phi = &particle.get_phi(self.name)

        # Host element
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self.name)

        cdef DTYPE_INT_t i

        for i in xrange(N_VERTICES):
            vertex = self.nv[i, host_element]
            var_last = var_last_arr[k, vertex]
            var_next = var_next_arr[k, vertex]

            if var_last != var_next:
                var_nodes[i] = interp.linear_interp(time_fraction, var_last, var_next)
            else:
                var_nodes[i] = var_last

            phi[i] = _phi.at(i)

        return interp.interpolate_within_element(var_nodes, phi)

    cdef void interpolate_grad_in_time_and_space(self, const DTYPE_FLOAT_t[:, ::1] &var_last_arr,
                                                 const DTYPE_FLOAT_t[:, ::1] &var_next_arr, DTYPE_INT_t k,
                                                 DTYPE_FLOAT_t time_fraction, Particle* particle, DTYPE_FLOAT_t var_prime[2]) except *:
        """ Interpolate the gradient in the given field in time and space

        Interpolate the gradient in the given field in time and space on the horizontal grid. The supplied fields
        should be 2D arrays of values defined at element nodes.

        Parameters
        ----------
        var_last_arr : 2D MemoryView
            Array of variable values at the last time index.

        var_next_arr : 2D MemoryView
            Array of variable values at the next time index.

        k : int
            k layer index.

        time_fraction : float
            Time interpolation coefficient

        particle: *Particle
            Pointer to a Particle object.

        var_prime : C array, float
            dvar_dx and dvar_dy components stored in a C array of length two.

        """
        # Gradients in phi
        cdef DTYPE_FLOAT_t dphi_dx[3]
        cdef DTYPE_FLOAT_t dphi_dy[3]

        cdef DTYPE_INT_t vertex # Vertex identifier

        # Intermediate values
        cdef DTYPE_FLOAT_t var_last, var_next
        cdef DTYPE_FLOAT_t var_nodes[3]

        # Host element
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self.name)

        cdef DTYPE_INT_t i

        # Get gradient in phi
        self.get_grad_phi(host_element, dphi_dx, dphi_dy)

        for i in xrange(N_VERTICES):
            vertex = self.nv[i, host_element]
            var_last = var_last_arr[k, vertex]
            var_next = var_next_arr[k, vertex]

            if var_last != var_next:
                var_nodes[i] = interp.linear_interp(time_fraction, var_last, var_next)
            else:
                var_nodes[i] = var_last

        # Interpolate d{}/dx and d{}/dy within the host element
        var_prime[0] = interp.interpolate_within_element(var_nodes, dphi_dx)
        var_prime[1] = interp.interpolate_within_element(var_nodes, dphi_dy)

        return

    cdef DTYPE_FLOAT_t shepard_interpolation(self, const DTYPE_FLOAT_t &x,
            const DTYPE_FLOAT_t &y, const DTYPE_FLOAT_t xpts[4], const DTYPE_FLOAT_t ypts[4],
            const DTYPE_FLOAT_t vals[4], const DTYPE_INT_t valid_points[4]) except FLOAT_ERR:
        """ Shepard interpolation in cartesian coordinates

        Distances are euclidian distances in the plane

        Parameters
        ----------
        x : float
            x-coordinate of the point at which data will be interpolated in m

        y : float
            y-coordinate of the point at which data will be interpolated in m

        xpts : C array, float
            x-coordinates of the points at which we have data (in m)

        ypts : C array, float
            y-coordinates of the points at which we have data (in m)

        vals : C array, float
            Values at the points where data is specified

        valid_points : C array, int
            Flags signifying which points to use (1) and not use (0).

        Returns
        -------
         : float
             The interpolated value.
        """
        # Euclidian distance between the point and a reference point
        cdef DTYPE_FLOAT_t r

        # Weighting applied to a given point
        cdef DTYPE_FLOAT_t w

        # Summed quantities
        cdef DTYPE_FLOAT_t sum
        cdef DTYPE_FLOAT_t sumw

        # For looping
        cdef int i

        # Loop over all reference points
        sum = 0.0
        sumw = 0.0
        for i in xrange(4):
            if valid_points[i] != 0:
                r = interp.get_euclidian_distance(x, y, xpts[i], ypts[i])
                if r == 0.0: return vals[i]
                w = 1.0/(r*r) # hardoced p value of -2
                sum = sum + w
                sumw = sumw + w*vals[i]

        return sumw/sum

cdef class UnstructuredGeographicGrid(Grid):
    """ Unstructured geographic grid

    Objects of type UnstructuredGeographicGrid can perform grid searches,
    compute local coordinates to assist with interpolation and help identify grid
    boundary crossings for unstructured geographic (lat/lon) grids.

    Parameters
    ----------
    config : configparser.ConfigParser
        PyLag configuration object.

    name : str
        The grid name. Useful if data are defined on multiple unstructured grids
        and a means is required to distinguish one from the other.

    n_nodes : int
        The number of nodes

    n_elems : int
        The number of elements

    nv : memoryview
        Memory view of nodes surrounding elements. With shape [3, n_elems]

    nbe : memoryview
        Memory view of elements surrounding elements. With shape [3, n_elems]

    x : 1D memory view
        x-coordinates of grid nodes in degrees longitude.

    y : 1D memory view
        y-coordinates of grid nodes in degrees latitude.

    xc : 1D memory view
        x-coordinates of element centres in degrees longitude.

    yc : 1D memory view
        y-coordinates of element centres in degrees latitude.

    land_sea_mask_c : 1D memory view
        Land sea element mask

    land_sea_mask : 1D memory view
        Land sea element mask at nodes
    """
    # Configurtion object
    cdef object config

    # The grid name
    cdef string name

    # Grid dimensions
    cdef DTYPE_INT_t n_elems, n_nodes

    # Element connectivity
    cdef DTYPE_INT_t[:,::1] nv

    # Element adjacency
    cdef DTYPE_INT_t[:,::1] nbe

    # Nodal coordinates in geographic coordinates. NB lon/lat in radians.
    cdef DTYPE_FLOAT_t[::1] lon_nodes
    cdef DTYPE_FLOAT_t[::1] lat_nodes
    cdef DTYPE_FLOAT_t[::1] r_nodes

    # Nodal coordinates in Cartesian coordinates
    cdef DTYPE_FLOAT_t[:,::1] points_nodes

    # Nodal coordinates in geographic coordinates. NB lon/lat in radians.
    cdef DTYPE_FLOAT_t[::1] lon_centres
    cdef DTYPE_FLOAT_t[::1] lat_centres
    cdef DTYPE_FLOAT_t[::1] r_centres

    # Element centre coordinates in Cartesian coordinates
    cdef DTYPE_FLOAT_t[:,::1] points_centres

    # Land sea element mask
    cdef DTYPE_INT_t[::1] land_sea_mask_c
    cdef DTYPE_INT_t[::1] land_sea_mask

    # Barycentric gradients
    cdef DTYPE_INT_t[:] barycentric_gradients_have_been_cached
    cdef DTYPE_FLOAT_t[:, ::1] dphi_dx
    cdef DTYPE_FLOAT_t[:, ::1] dphi_dy

    # Element areas
    cdef DTYPE_FLOAT_t[::1] areas

    def __init__(self, config, name, n_nodes, n_elems, nv, nbe, x, y, xc, yc, land_sea_mask_c, land_sea_mask, areas=None):

        self.config = config

        self.name = name

        # Grid structure
        self.n_nodes = n_nodes
        self.n_elems = n_elems
        self.nv = nv
        self.nbe = nbe

        # Nodal coordinates (geographic coordinates)
        self.lon_nodes = x
        self.lat_nodes = y
        self.r_nodes = np.ones(n_nodes, dtype=DTYPE_FLOAT)

        # Nodel coordinates (cartesian coordinates)
        x_nodes, y_nodes, z_nodes = geographic_to_cartesian_coords_python(x, y)
        self.points_nodes = np.column_stack((x_nodes, y_nodes, z_nodes))

        # Element centre coordinates (geographic coordinates)
        self.lon_centres = xc
        self.lat_centres = yc
        self.r_centres = np.ones(n_elems, dtype=DTYPE_FLOAT)

        # Element centre coordinates (cartesian coordinates)
        x_centres, y_centres, z_centres = geographic_to_cartesian_coords_python(xc, yc)
        self.points_centres = np.column_stack((x_centres, y_centres, z_centres))

        # Masks
        self.land_sea_mask_c = land_sea_mask_c[:]
        self.land_sea_mask = land_sea_mask[:]

        # Element areas
        if areas is not None:
            self.areas = areas

        # Containers for preserving the value of gradient calculations
        self.barycentric_gradients_have_been_cached = np.zeros(self.n_elems, dtype=DTYPE_INT, order='C')
        self.dphi_dx = np.ones((self.n_elems, 3), dtype=DTYPE_FLOAT, order='C') * -999.
        self.dphi_dy = np.ones((self.n_elems, 3), dtype=DTYPE_FLOAT, order='C') * -999.

    cdef DTYPE_INT_t find_host_using_local_search(self, Particle *particle) except INT_ERR:
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

        Parameters
        ----------
        particle: *Particle
            The particle.

        Returns
        -------
        flag : int
            Integer flag that indicates whether or not the seach was successful.
        """
        # Intermediate arrays/variables
        cdef vector[DTYPE_FLOAT_t] phi
        cdef DTYPE_FLOAT_t s[3]
        cdef DTYPE_FLOAT_t s_test

        cdef bint host_found

        cdef DTYPE_INT_t n_host_land_boundaries

        cdef DTYPE_INT_t flag, guess, last_guess, second_to_last_guess

        # Check for non-sensical start points.
        guess = particle.get_host_horizontal_elem(self.name)
        if guess < 0:
            raise ValueError('Invalid start point for local host element '\
                    'search. Start point = {}'.format(guess))

        host_found = False
        last_guess = -1
        second_to_last_guess = -1

        while True:
            # Tetrahedral coordinates
            self.get_tetrahedral_coords(particle.get_x1(), particle.get_x2(), guess, s)

            # Check to see if the particle is in the current element
            s_test = float_min(float_min(s[0], s[1]), s[2])
            if s_test >= 0.0:
                host_found = True

            # Check to see if the particle has walked into an invalid element (e.g. an
            # element treated as land)
            if host_found is True:
                if self.land_sea_mask_c[guess] != LAND:
                    # Normal element
                    particle.set_host_horizontal_elem(self.name, guess)

                    phi = self.get_normalised_tetrahedral_coords(s)

                    particle.set_phi(self.name, phi)

                    return IN_DOMAIN
                else:
                    return IN_MASKED_ELEM

            # If not, use phi to select the next element to be searched
            second_to_last_guess = last_guess
            last_guess = guess
            if s[0] == s_test:
                guess = self.nbe[0,last_guess]
            elif s[1] == s_test:
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
        cdef int i # Loop counter
        cdef int vertex # Vertex identifier
        cdef DTYPE_INT_t elem, last_elem, current_elem # Element identifies
        cdef DTYPE_INT_t flag, host

        # Intermediate arrays/variables
        cdef DTYPE_FLOAT_t x_tri[3]
        cdef DTYPE_FLOAT_t y_tri[3]

        # 2D position vectors for the end points of the element's side
        cdef DTYPE_FLOAT_t x1[2]
        cdef DTYPE_FLOAT_t x2[2]

        # 2D position vectors for the particle's previous and new position
        cdef DTYPE_FLOAT_t x3[2]
        cdef DTYPE_FLOAT_t x4[2]

        # Containers for tetrahedral coordinates
        cdef DTYPE_FLOAT_t s[3]
        cdef DTYPE_FLOAT_t s_test

        # Normalalised tetrahedral coordinates
        cdef vector[DTYPE_FLOAT_t] phi

        # Intermediate arrays
        cdef DTYPE_INT_t x1_indices[3]
        cdef DTYPE_INT_t x2_indices[3]
        cdef DTYPE_INT_t nbe_indices[3]

        # Array indices
        cdef int x1_idx
        cdef int x2_idx
        cdef int nbe_idx

        # Initialise arrays
        x1_indices[:] = [0,1,2]
        x2_indices[:] = [1,2,0]
        nbe_indices[:] = [2,0,1]

        # Construct arrays to hold the coordinates of the particle's previous
        # position vector and its new position vector
        x3[0] = particle_old.get_x1(); x3[1] = particle_old.get_x2()
        x4[0] = particle_new.get_x1(); x4[1] = particle_new.get_x2()

        # Start the search using the host known to contain (x1_old, x2_old)
        elem = particle_old.get_host_horizontal_elem(self.name)

        # Set last_elem equal to elem in the first instance
        last_elem = elem

        while True:
            # Extract nodal coordinates
            for i in xrange(3):
                vertex = self.nv[i,elem]
                x_tri[i] = self.lon_nodes[vertex]
                y_tri[i] = self.lat_nodes[vertex]

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

                if great_circle_arc_segments_intersect(x1, x2, x3, x4) == 0:
                    # Lines do not intersect - check the next one
                    continue

                # Intersection found - keep a record of the last element checked
                last_elem = elem

                # Index for the neighbour element
                elem = self.nbe[nbe_idx, elem]

                # Check to see if the pathline has exited the domain
                if elem >= 0:
                    if self.land_sea_mask_c[elem] == LAND:
                        flag = LAND_BDY_CROSSED

                        # Set host to the last element the particle passed through
                        particle_new.set_host_horizontal_elem(self.name, last_elem)
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
                    particle_new.set_host_horizontal_elem(self.name, last_elem)

                    return flag

            if current_elem == elem:
                # The algorithm indicates the particle has not exited the current element; we
                # check the tetrahedral coordinates to see if they are indeed all +ve.
                self.get_tetrahedral_coords(particle_new.get_x1(), particle_new.get_x2(), current_elem, s)
                s_test = float_min(float_min(s[0], s[1]), s[2])
                if s_test >= 0.0:
                    # Methods agree. Flag particle as being in the domain.
                    particle_new.set_host_horizontal_elem(self.name, current_elem)
                    phi = self.get_normalised_tetrahedral_coords(s)
                    particle_new.set_phi(self.name, phi)
                else:
                    # Methods disagree. This can happen due to numerical precision issues. Respond
                    # by moving the particle to a default location.
                    self.set_default_location(particle_new)
                flag = IN_DOMAIN

                return flag

    cdef DTYPE_INT_t find_host_using_global_search(self, Particle *particle) except INT_ERR:
        """ Returns the host horizontal element through global searching.

        Sequentially search all elements for the given location. Set the particle
        host element if found.

        Parameters
        ----------
        particle_old: *Particle
            The particle.

        Returns
        -------
        flag : int
            Integer flag that indicates whether or not the seach was successful.
        """
        # Intermediate arrays/variables
        cdef vector[DTYPE_FLOAT_t] phi = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)
        cdef DTYPE_FLOAT_t s[3]
        cdef DTYPE_FLOAT_t s_test

        cdef bint host_found

        cdef DTYPE_INT_t n_host_land_boundaries

        cdef DTYPE_INT_t i, guess

        host_found = False

        for guess in xrange(self.n_elems):
            # Barycentric coordinates
            self.get_tetrahedral_coords(particle.get_x1(), particle.get_x2(), guess, s)

            # Check to see if the particle is in the current element
            s_test = float_min(float_min(s[0], s[1]), s[2])
            if s_test >= 0.0:
                host_found = True

            if host_found is True:
                if self.land_sea_mask_c[guess] != LAND:
                    particle.set_host_horizontal_elem(self.name, guess)

                    phi = self.get_normalised_tetrahedral_coords(s)

                    particle.set_phi(self.name, phi)

                    return IN_DOMAIN
                else:
                    return BDY_ERROR
        return BDY_ERROR

    cdef get_boundary_intersection(self,
                                   Particle *particle_old,
                                   Particle *particle_new,
                                   DTYPE_FLOAT_t start_point[2],
                                   DTYPE_FLOAT_t end_point[2],
                                   DTYPE_FLOAT_t intersection[2]):
        """ Find the boundary intersection point

        This function is primarily intended to assist in the application of
        horizontal boundary conditions where it is often necessary to establish
        the point on a side of an element at which a particle crossed before
        exiting the model domain.

        Parameters
        ----------
        particle_old: *Particle
            The particle at its old position.

        particle_new: *Particle
            The particle at its new position.

        start_point : C array, [float, float]
            Start coordinates of the side the particle crossed.

        end_point : C array, [float, float]
            End coordinates of the side the particle crossed.

        intersection : C array [float, float]
            Coordinates of the intersection point.

        Returns
        -------
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

        # Initialise arrays
        x1_indices[:] = [0,1,2]
        x2_indices[:] = [1,2,0]
        nbe_indices[:] = [2,0,1]

        # Construct arrays to hold the coordinates of the particle's previous
        # position vector and its new position vector
        x3[0] = particle_old.get_x1(); x3[1] = particle_old.get_x2()
        x4[0] = particle_new.get_x1(); x4[1] = particle_new.get_x2()

        # Extract nodal coordinates
        for i in xrange(3):
            vertex = self.nv[i, particle_new.get_host_horizontal_elem(self.name)]
            x_tri[i] = self.lon_nodes[vertex]
            y_tri[i] = self.lat_nodes[vertex]

        # Loop over all sides of the element to find the land boundary the element crossed
        for i in xrange(3):
            x1_idx = x1_indices[i]
            x2_idx = x2_indices[i]
            nbe_idx = nbe_indices[i]

            nbe = self.nbe[nbe_idx, particle_new.get_host_horizontal_elem(self.name)]
            if nbe != -1:
                # Masked elements are treated as land too. If the neighbour isn't masked, continue the search.
                if self.land_sea_mask_c[nbe] != LAND:
                    continue

            # End coordinates for the side
            x1[0] = x_tri[x1_idx]; x1[1] = y_tri[x1_idx]
            x2[0] = x_tri[x2_idx]; x2[1] = y_tri[x2_idx]

            if get_intersection_point_in_geographic_coordinates(x1, x2, x3, x4, xi) == 1:
                for i in range(2):
                    start_point[i] = x1[i]
                    end_point[i] = x2[i]
                    intersection[i] = xi[i]
                return
            else:
                continue

        raise RuntimeError('Failed to calculate boundary intersection.')

    cdef set_default_location(self, Particle *particle):
        """ Set default location

        Move the particle to its host element's centroid.
        """
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self.name)

        particle.set_x1(self.lon_centres[host_element])
        particle.set_x2(self.lat_centres[host_element])
        self.set_local_coordinates(particle)
        return

    cdef set_local_coordinates(self, Particle *particle):
        """ Set local coordinates

        Each particle has associated with it a set of global coordinates
        and a set of local coordinates. Here, the global coordinates and the
        host horizontal element are used to set the local coordinates.

        Parameters
        ----------
        particle: *Particle
            Pointer to a Particle struct
        """
        cdef vector[DTYPE_FLOAT_t] phi

        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self.name)

        cdef DTYPE_INT_t i

        phi = self.get_phi(particle.get_x1(), particle.get_x2(), host_element)

        # Check for negative values.
        for i in xrange(3):
            if phi[i] >= 0.0:
                continue
            elif phi[i] >= -EPSILON:
                phi[i] = 0.0
            else:
                s = to_string(particle)
                msg = "One or more local coordinates are invalid (phi = {}) \n\n"\
                      "The following information may be used to study the \n"\
                      "failure in more detail. \n\n"\
                      "{}".format(phi[i], s)
                print(msg)

                raise ValueError('One or more local coordinates are negative')

        # Set phi
        particle.set_phi(self.name, phi)

    cdef DTYPE_FLOAT_t get_element_area(self, Particle *particle) except FLOAT_ERR:
        """ Return the element's area within which the particle resides

        Parameters
        ----------
        particle: *Particle
            Pointer to a Particle object
        """
        # Loop counter
        cdef int i

        # Vertex identifiers
        cdef DTYPE_INT_t vertex_0, vertex_1, vertex_2

        # The host element
        cdef DTYPE_INT_t host

        # Element vertex coordinates in cartesian coordinates
        cdef DTYPE_FLOAT_t p0[3]
        cdef DTYPE_FLOAT_t p1[3]
        cdef DTYPE_FLOAT_t p2[3]

        # Element area
        cdef DTYPE_FLOAT_t area

        # Get the host
        host = particle.get_host_horizontal_elem(self.name)

        if self.areas is not None:
            area = self.areas[host]
        else:
            # Extract nodal cartesian coordinates
            vertex_0 = self.nv[0, host]
            vertex_1 = self.nv[1, host]
            vertex_2 = self.nv[2, host]
            for i in range(3):
                p0[i] = self.points_nodes[vertex_0, i]
                p1[i] = self.points_nodes[vertex_1, i]
                p2[i] = self.points_nodes[vertex_2, i]

            area = area_of_a_spherical_triangle(p0, p1, p2, earth_radius)

        return area

    cpdef vector[DTYPE_FLOAT_t] get_phi(self, const DTYPE_FLOAT_t &x1, const DTYPE_FLOAT_t &x2, const DTYPE_INT_t &host):
        """ Get normalised tetrahedral coordinates given a point's position and the host element

        Parameters
        ----------
        x1 : float
            Longitude in radians

        x2 : float
            Latitude in radians

        host : int
            Host element

        Returns
        -------
        phi : vector[FLOAT]
            Three vector giving a point's normalised tetrahedral coordinates within a spherical triangle.
        """
        cdef DTYPE_FLOAT_t s[3]

        self.get_tetrahedral_coords(x1, x2, host, s)

        return self.get_normalised_tetrahedral_coords(s)

    cdef vector[DTYPE_FLOAT_t] get_normalised_tetrahedral_coords(self, const DTYPE_FLOAT_t s[3]):
        """ Get normalised tetrahedral coordinates given the tetrahedral coordinates

        Parameters
        ----------
        s : C array, [FLOAT]
            Three vector giving a point's tetrahedral coordinates within a spherical triangle.

        Returns
        -------
        phi : vector[FLOAT]
            Three vector giving a point's normalised tetrahedral coordinates within a spherical triangle.
        """
        cdef vector[DTYPE_FLOAT_t] phi = vector[DTYPE_FLOAT_t](N_VERTICES, -999.)
        cdef DTYPE_FLOAT_t s_sum

        s_sum = s[0] + s[1] + s[2]

        phi[0] = s[0] / s_sum
        phi[1] = s[1] / s_sum
        phi[2] = s[2] / s_sum

        return phi

    cdef get_tetrahedral_coords(self, const DTYPE_FLOAT_t &x1, const DTYPE_FLOAT_t &x2, const DTYPE_INT_t &host,
                                DTYPE_FLOAT_t s[3]):
        """ Get tetrahedral coordinates given a point's position and the host element

        The method uses the approach described by Lawson (1984).

        Parameters
        ----------
        x1 : float
            Particle longitude in radians.

        x2 : float
            Particle latitude in radians.

        host : int
            Host element.

        s : C array, [float, float, float]
            Vector container in which the tetrahedral coordinates are saved.

        References
        ----------
        1) Lawson, C. 1984, C1 Surface interpolation for scattered data on the surface
        of a sphere. Rocky mountain journal of mathematics. 14:1.
        """

        # Loop counter
        cdef int i

        # Vertex identifiers
        cdef DTYPE_INT_t vertex_0, vertex_1, vertex_2

        # Element vertex coordinates in cartesian coordinates
        cdef DTYPE_FLOAT_t p[3]
        cdef DTYPE_FLOAT_t p0[3]
        cdef DTYPE_FLOAT_t p1[3]
        cdef DTYPE_FLOAT_t p2[3]

        # x1 and x2 in radians
        cdef DTYPE_FLOAT_t x1_rad, x2_rad

        # Extract nodal cartesian coordinates
        vertex_0 = self.nv[0, host]
        vertex_1 = self.nv[1, host]
        vertex_2 = self.nv[2, host]
        for i in range(3):
            p0[i] = self.points_nodes[vertex_0, i]
            p1[i] = self.points_nodes[vertex_1, i]
            p2[i] = self.points_nodes[vertex_2, i]

        # For the supplied point only, convert to cartesian coordinates on the unit sphere
        geographic_to_cartesian_coords(x1, x2, 1.0, p)

        # Compute tetrahedral coordinates (NB clockwise point ordering)
        s[0] = det_third_order(p, p2, p1)
        s[1] = det_third_order(p2, p, p0)
        s[2] = det_third_order(p1, p0, p)

        # Flush small values to zero
        for i in range(3):
            if s[i] < 0.0 and s[i] >= -EPSILON:
                s[i] = 0.0

    cdef void get_grad_phi(self, DTYPE_INT_t host,
                           DTYPE_FLOAT_t dphi_dx[3],
                           DTYPE_FLOAT_t dphi_dy[3]) except *:
        """ Get gradient in phi with respect to x and y

        The gradient in phi is calculated by first converting to cartesian
        coordinates, then rotating the cartesian axes so that the positive
        z-axis forms an outward normal through the element's centroid, while
        the x- and y- axes are locally aligned with lines of constant longitude
        and latitude respectively. The element is then projected onto the plane
        that lies tangential to the surface of the sphere at the element's
        centroid. The gradients dphi/dx and dphi/dy are then computed
        using the formulas:

        dphi_i/dx = \Delta y_i / 2A
        dphi_i/dy = - \Delta_x_i / 2A

        Where A is the element's area, given by the equation:

        A = sum(i=1,3) x_i * \Delta y_i

        The Delta terms are simply the difference between the x or y coordinates of the
        element vertices other than that referred to by i (see Lynch 2015, p233).
        As dpi_i/dx and dphi_i/dy are constant over the element, they are the same
        for all particles in the element. To save repeated calculations, gradient
        terms are cached as the simulation proceeds with cached values returned after
        the first request.

        Parameters
        ----------
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
        cdef DTYPE_FLOAT_t x_tri[3]
        cdef DTYPE_FLOAT_t y_tri[3]
        cdef DTYPE_FLOAT_t xc, yc

        # Element cartesian coordinates
        cdef DTYPE_FLOAT_t pc[3]
        cdef DTYPE_FLOAT_t p0[3]
        cdef DTYPE_FLOAT_t p1[3]
        cdef DTYPE_FLOAT_t p2[3]

        # Rotated element cartesian coordinates
        cdef DTYPE_FLOAT_t pc_rot[3]
        cdef DTYPE_FLOAT_t p0_rot[3]
        cdef DTYPE_FLOAT_t p1_rot[3]
        cdef DTYPE_FLOAT_t p2_rot[3]

        cdef DTYPE_FLOAT_t a1, a2, a3, a4, twice_signed_element_area

        # Return cached values if they have been pre-computed
        if self.barycentric_gradients_have_been_cached[host] == 1:
            for i in range(3):
                dphi_dx[i] = self.dphi_dx[host, i]
                dphi_dy[i] = self.dphi_dy[host, i]
            return

        # Element vertices
        for i in xrange(3):
            vertex = self.nv[i,host]
            x_tri[i] = self.lon_nodes[vertex]
            y_tri[i] = self.lat_nodes[vertex]

        # The element centroid in radians
        xc = self.lon_centres[host]
        yc = self.lat_centres[host]

        # Convert to cartesian coordinates
        geographic_to_cartesian_coords(xc, yc, earth_radius, pc)
        geographic_to_cartesian_coords(x_tri[0], y_tri[0], earth_radius, p0)
        geographic_to_cartesian_coords(x_tri[1], y_tri[1], earth_radius, p1)
        geographic_to_cartesian_coords(x_tri[2], y_tri[2], earth_radius, p2)

        # Rotate axes to get the desired orientation as described above
        rotate_axes(pc, xc, yc, pc_rot)
        rotate_axes(p0, xc, yc, p0_rot)
        rotate_axes(p1, xc, yc, p1_rot)
        rotate_axes(p2, xc, yc, p2_rot)

        # Now calculate gradients within the projected planar triangle

        # Intermediate terms
        a1 = p1_rot[0] - p0_rot[0]
        a2 = p2_rot[1] - p0_rot[1]
        a3 = p1_rot[1] - p0_rot[1]
        a4 = p2_rot[0] - p0_rot[0]

        # Evaluate the vector cross product
        twice_signed_element_area = a1 * a2 - a3 * a4

        dphi_dx[0] = (p1_rot[1] - p2_rot[1])/twice_signed_element_area
        dphi_dx[1] = (p2_rot[1] - p0_rot[1])/twice_signed_element_area
        dphi_dx[2] = (p0_rot[1] - p1_rot[1])/twice_signed_element_area

        dphi_dy[0] = (p2_rot[0] - p1_rot[0])/twice_signed_element_area
        dphi_dy[1] = (p0_rot[0] - p2_rot[0])/twice_signed_element_area
        dphi_dy[2] = (p1_rot[0] - p0_rot[0])/twice_signed_element_area

        # Cache values
        self.barycentric_gradients_have_been_cached[host] = 1
        for i in range(3):
            self.dphi_dx[host, i] = dphi_dx[i]
            self.dphi_dy[host, i] = dphi_dy[i]

        return

    cdef DTYPE_FLOAT_t interpolate_in_space(self, DTYPE_FLOAT_t[::1] var_arr,  Particle* particle) except FLOAT_ERR:
        """ Interpolate the given field in space

        Interpolate the given field on the horizontal grid. The supplied fields
        should be 1D arrays of values defined at element nodes.

        Parameters
        ----------
        var_last_arr : 1D MemoryView
            Array of variable values at the last time index.

        var_next_arr : 1D MemoryView
            Array of variable values at the next time index.

        particle: *Particle
            Pointer to a Particle object.

        Returns
        -------
         : float
            The interpolated value of the variable
        """
        cdef DTYPE_INT_t vertex # Vertex identifier
        cdef DTYPE_FLOAT_t var_nodes[3]
        cdef DTYPE_FLOAT_t phi[3]

        # Phi
        cdef const vector[DTYPE_FLOAT_t] *_phi = &particle.get_phi(self.name)

        # Host element
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self.name)

        cdef DTYPE_INT_t i

        for i in xrange(N_VERTICES):
            vertex = self.nv[i, host_element]
            var_nodes[i] = var_arr[vertex]
            phi[i] = _phi.at(i)

        if self.land_sea_mask_c[host_element] == SEA:
            # Normal sea element
            return interp.interpolate_within_element(var_nodes, phi)

        elif self.land_sea_mask_c[host_element] == BOUNDARY_ELEMENT:
            # Boundary element with masked nodes. Adjust interpolation coefficients.
            self._adjust_interpolation_coefficients(host_element, phi)
            return interp.interpolate_within_element(var_nodes, phi)

        else:
            raise RuntimeError('Cannot interpolate within masked element `{}`.'.format(self.land_sea_mask_c[host_element]))

    cdef DTYPE_FLOAT_t interpolate_in_time_and_space_2D(self, DTYPE_FLOAT_t[::1] var_last_arr,
                                                        DTYPE_FLOAT_t[::1] var_next_arr,
                                                        DTYPE_FLOAT_t time_fraction, Particle* particle) except FLOAT_ERR:
        """ Interpolate the given field in time and space 2D

        Interpolate the given field in time and space on the horizontal grid. The supplied fields
        should be 1D arrays of values defined at element nodes.

        Parameters
        ----------
        var_last_arr : 1D MemoryView
            Array of variable values at the last time index.

        var_next_arr : 1D MemoryView
            Array of variable values at the next time index.

        time_fraction : float
            Time interpolation coefficient

        particle: *Particle
            Pointer to a Particle object.

        Returns
        -------
         : float
            The interpolated value of the variable
        """
        cdef DTYPE_INT_t vertex # Vertex identifier
        cdef DTYPE_FLOAT_t var_last, var_next
        cdef DTYPE_FLOAT_t var_nodes[3]
        cdef DTYPE_FLOAT_t phi[3]

        # Phi
        cdef const vector[DTYPE_FLOAT_t] *_phi = &particle.get_phi(self.name)

        # Host element
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self.name)

        cdef DTYPE_INT_t i

        for i in range(N_VERTICES):
            vertex = self.nv[i, host_element]
            var_last = var_last_arr[vertex]
            var_next = var_next_arr[vertex]

            if var_last != var_next:
                var_nodes[i] = interp.linear_interp(time_fraction, var_last, var_next)
            else:
                var_nodes[i] = var_last

            phi[i] = _phi.at(i)

        if self.land_sea_mask_c[host_element] == SEA:
            # Normal sea element
            return interp.interpolate_within_element(var_nodes, phi)

        elif self.land_sea_mask_c[host_element] == BOUNDARY_ELEMENT:
            # Boundary element with masked nodes. Adjust interpolation coefficients.
            self._adjust_interpolation_coefficients(host_element, phi)
            return interp.interpolate_within_element(var_nodes, phi)

        else:
            raise RuntimeError('Cannot interpolate within masked element `{}`.'.format(self.land_sea_mask_c[host_element]))

    cdef DTYPE_FLOAT_t interpolate_in_time_and_space(self, DTYPE_FLOAT_t[:, ::1] var_last_arr,
                                                     DTYPE_FLOAT_t[:, ::1] var_next_arr, DTYPE_INT_t k,
                                                     DTYPE_FLOAT_t time_fraction, Particle* particle) except FLOAT_ERR:
        """ Interpolate the given field in time and space

        Interpolate the given field in time and space on the horizontal grid. The supplied fields
        should be 2D arrays of values defined at element nodes.

        Parameters
        ----------
        var_last_arr : 2D MemoryView
            Array of variable values at the last time index.

        var_next_arr : 2D MemoryView
            Array of variable values at the next time index.

        k : int
            The depth layer index

        time_fraction : float
            Time interpolation coefficient

        particle: *Particle
            Pointer to a Particle object.

        Returns
        -------
         : float
            The interpolated value of the variable
        """
        cdef DTYPE_INT_t vertex # Vertex identifier
        cdef DTYPE_FLOAT_t var_last, var_next
        cdef DTYPE_FLOAT_t var_nodes[3]
        cdef DTYPE_FLOAT_t phi[3]

        # Phi
        cdef const vector[DTYPE_FLOAT_t] *_phi = &particle.get_phi(self.name)

        # Host element
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self.name)

        cdef DTYPE_INT_t i

        for i in range(N_VERTICES):
            vertex = self.nv[i, host_element]
            var_last = var_last_arr[k, vertex]
            var_next = var_next_arr[k, vertex]

            if var_last != var_next:
                var_nodes[i] = interp.linear_interp(time_fraction, var_last, var_next)
            else:
                var_nodes[i] = var_last

            phi[i] = _phi.at(i)

        if self.land_sea_mask_c[host_element] == SEA:
            # Normal sea element
            return interp.interpolate_within_element(var_nodes, phi)

        elif self.land_sea_mask_c[host_element] == BOUNDARY_ELEMENT:
            # Boundary element with masked nodes. Adjust interpolation coefficients.
            self._adjust_interpolation_coefficients(host_element, phi)
            return interp.interpolate_within_element(var_nodes, phi)

        else:
            raise RuntimeError('Cannot interpolate within masked element `{}`.'.format(self.land_sea_mask_c[host_element]))


    cdef void _adjust_interpolation_coefficients(self, const DTYPE_INT_t host,
                                                 DTYPE_FLOAT_t phi[3]) except *:
        """ Adjust interpolation coefficients

        Adjust the interpolation coefficients so that a nearest neighbour
        algorithm is used. The algorithm sets the value of phi for the
        nearest node to 1.0 and all other entries to 0.0. The method can
        be useful when interpolating within boundary elements with masked
        nodes as it prevents masked nodes with missing values being used to
        compute the value of a given variable inside the element.

        Parameters
        ----------
        host : int
            The host element.

        phi : C array, float
            Interpolation coefficients in the host element.
        """
        cdef DTYPE_FLOAT_t phi_new[N_VERTICES]
        cdef DTYPE_FLOAT_t phi_test
        cdef DTYPE_INT_t index
        cdef DTYPE_INT_t node
        cdef DTYPE_INT_t i

        # Initialise all phi's to zero
        phi_test = 0.0
        for i in range(N_VERTICES):
            phi_new[i] = 0.0

        # Try to find the index of the nearest neighbour
        index = INT_ERR
        for i in range(N_VERTICES):
            node = self.nv[i, host]
            if self.land_sea_mask[node] == SEA:
                if phi[i] > phi_test:
                    phi_test = phi[i]
                    index = i

        if index == INT_ERR:
            # The nearest neighbour wasn't found, meaning there were no sea
            # points which a value of phi > 0.0. Although unlikely, this could
            # happen if the particle was sat on a masked node. In this case, we search
            # for an unmasked node and use it as the nearest neighbour.
            for i in range(N_VERTICES):
                node = self.nv[i, host]
                if self.land_sea_mask[node] == SEA:
                    index = i
                    break

        if index == INT_ERR:
            # If no index has been found, then all sea points must be masked.
            raise PyLagRuntimeError(f'All nodes of element {host} are masked!')

        # Nearest neighbour was found - set the corresponding phi to 1.0
        phi_new[index] = 1.0

        # Copy phi_new into phi
        for i in range(N_VERTICES):
            phi[i] = phi_new[i]

    cdef void interpolate_grad_in_time_and_space(self, const DTYPE_FLOAT_t[:, ::1] &var_last_arr,
                                                 const DTYPE_FLOAT_t[:, ::1] &var_next_arr, DTYPE_INT_t k,
                                                 DTYPE_FLOAT_t time_fraction, Particle* particle,
                                                 DTYPE_FLOAT_t var_prime[2]) except *:
        """ Interpolate the gradient in the given field in time and space

        Interpolate the gradient in the given field in time and space on the horizontal grid. The supplied fields
        should be 2D arrays of values defined at element nodes.

        Parameters
        ----------
        var_last_arr : 2D MemoryView
            Array of variable values at the last time index.

        var_next_arr : 2D MemoryView
            Array of variable values at the next time index.

        k : int
            The depth layer index

        time_fraction : float
            Time interpolation coefficient

        particle: *Particle
            Pointer to a Particle object.

        var_prime : C array, float
            dvar_dx and dvar_dy components stored in a C array of length two.

        """
        # Gradients in phi
        cdef DTYPE_FLOAT_t dphi_dx[3]
        cdef DTYPE_FLOAT_t dphi_dy[3]

        cdef DTYPE_INT_t vertex # Vertex identifier

        # Intermediate values
        cdef DTYPE_FLOAT_t var_last, var_next
        cdef DTYPE_FLOAT_t var_nodes[3]

        # Host element
        cdef DTYPE_INT_t host_element = particle.get_host_horizontal_elem(self.name)

        cdef DTYPE_INT_t i

        # Get gradient in phi
        self.get_grad_phi(host_element, dphi_dx, dphi_dy)

        for i in xrange(N_VERTICES):
            vertex = self.nv[i, host_element]
            var_last = var_last_arr[k, vertex]
            var_next = var_next_arr[k, vertex]

            if var_last != var_next:
                var_nodes[i] = interp.linear_interp(time_fraction, var_last, var_next)
            else:
                var_nodes[i] = var_last

        # Interpolate d{}/dx and d{}/dy within the host element
        var_prime[0] = interp.interpolate_within_element(var_nodes, dphi_dx)
        var_prime[1] = interp.interpolate_within_element(var_nodes, dphi_dy)

        return

    cdef DTYPE_FLOAT_t shepard_interpolation(self, const DTYPE_FLOAT_t &x,
            const DTYPE_FLOAT_t &y, const DTYPE_FLOAT_t xpts[4], const DTYPE_FLOAT_t ypts[4],
            const DTYPE_FLOAT_t vals[4], const DTYPE_INT_t valid_points[4]) except FLOAT_ERR:
        """ Shepard interpolation in geographic coordinates

        Distances are here calculated as segments of great circles joining two points
        on the sphere.

        Parameters
        ----------
        x : float
            Longitude of the point at which data will be interpolated in radians.

        y : float
            Latitude of the point at which data will be interpolated in radians.

        xpts : C array, float
            Longitudes of the points at which we have data in radians.

        ypts : C array, float
            Latitude of the points at which we have data in radians.

        vals : C array, float
            Values at the points where data is specified

        valid_points : C array, int
            Flags signifying which points to use (1) and not use (0).

        Returns
        -------
         : float
             The interpolated value.
        """
        # Great circle distance between the point and a reference point
        cdef DTYPE_FLOAT_t r

        # Weighting applied to a given point
        cdef DTYPE_FLOAT_t w

        # Summed quantities
        cdef DTYPE_FLOAT_t sum
        cdef DTYPE_FLOAT_t sumw

        # For looping
        cdef int i

        # Loop over all reference points
        sum = 0.0
        sumw = 0.0
        for i in xrange(4):
            if valid_points[i] != 0:
                r = haversine(x, y, xpts[i], ypts[i])
                if r == 0.0: return vals[i]
                w = 1.0/(r*r) # hardoced p value of -2
                sum = sum + w
                sumw = sumw + w*vals[i]

        return sumw / sum

def get_unstructured_grid(config, *args, **kwargs):
    """ Factory method for unstructured grid types

    The factory method is used to distinguish between geographic and cartesian
    unstructured grid types. Required arguments are documented in the respective
    class docs.

    Parameters
    ----------
    config : ConfigParser
        Object of type ConfigParser.
    """
    coordinate_system = config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system")
    if coordinate_system == "cartesian":
        return UnstructuredCartesianGrid(config, *args, **kwargs)
    elif coordinate_system == "geographic":
        return UnstructuredGeographicGrid(config, *args, **kwargs)
    else:
        raise ValueError("Unsupported coordinate_system `{}` specified".format(coordinate_system))
