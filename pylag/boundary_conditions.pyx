"""
A set of classes for handling particle interactions with horizontal and vertical
boundaries.

Note
----
boundary_conditions is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""
include "constants.pxi"

from libcpp.vector cimport vector

import logging

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

# PyLag cimports
from pylag.particle_cpp_wrapper cimport ParticleSmartPtr
from pylag.math cimport inner_product_two as inner_product
from pylag.math cimport rotate_axes, reverse_rotate_axes
from pylag.math cimport cartesian_to_geographic_coords
from pylag.math cimport geographic_to_cartesian_coords
from pylag.math cimport Intersection
from pylag.parameters cimport radians_to_deg


cdef class HorizBoundaryConditionCalculator:
    """ Base class for horizontal boundary condition calculators
    """

    def apply_wrapper(self, DataReader data_reader,
                      ParticleSmartPtr particle_old,
                      ParticleSmartPtr particle_new):
        """ Python friendly wrapper for apply()

        In the apply() method must be implemented in the derived class.
        It should apply the selected horizontal boundary condition in response
        to a particle crossing a lateral boundary, and update the position of
        `particle_new`.

        Parameters
        ----------
        data_reader : pylag.data_reader.DataReader
            A concrete PyLag data reader which inherits from the base class
            `pylag.data_reader.DataReader`.

        particle_old : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle before it crossed the lateral boundary. The particle
            should lie inside the model domain.

        particle_new : pylag.particle_cpp_Wrapper.ParticleSmartPtr
            The particle after it has crossed the lateral boundary. The particle
            should lie outside the model domain.

        Returns
        -------
         : int
            Flag signifying whether or not the application was successful.
        """
        return self.apply(data_reader, particle_old.get_ptr(), particle_new.get_ptr())

    cdef DTYPE_INT_t apply(self, DataReader data_reader, Particle *particle_old,
                           Particle *particle_new) except INT_ERR:
        raise NotImplementedError


cdef class RestoringHorizCartesianBoundaryConditionCalculator(HorizBoundaryConditionCalculator):
    """ Restoring horizontal boundary condition calculator for cartesian grids

    Restoring horizontal boundary condition calculators for cartesian coordinate
    grids simply move the particle back to its last valid position.
    """
    cdef DTYPE_INT_t apply(self, DataReader data_reader, Particle *particle_old,
                           Particle *particle_new) except INT_ERR:
        cdef DTYPE_INT_t flag

        # Move the particle back to its last known valid position
        particle_new.set_x1(particle_old.get_x1())
        particle_new.set_x2(particle_old.get_x2())
        flag = data_reader.find_host(particle_old, particle_new)

        if flag == IN_DOMAIN:
            return flag

        return BDY_ERROR


cdef class RefHorizCartesianBoundaryConditionCalculator(HorizBoundaryConditionCalculator):
    """ Reflecting horizontal boundary condition calculator for cartesian grids

    Reflecting horizontal boundary condition calculators reflect particles
    back into the domain should they have crossed a model boundary. A particle
    may be reflected multiple times if the first or a subsequent reflection also
    leaves it outside of the domain. The position of the reflected particle is
    updated in place.
    """

    cdef DTYPE_INT_t apply(self, DataReader data_reader, Particle *particle_old,
               Particle *particle_new) except INT_ERR:
        """Apply reflecting boundary conditions
        
        The algorithm computes reflection vectors for particles given
        information that describes their last known good position (which was
        inside the domain) and their new position which has been flagged as
        lying outside of the domain. It does this by first calculating where
        the particle pathline intersected the model boundary. The process is
        iterative, allowing for situations in which the reflected point
        is still outside the domain - this can happen in corners of the model
        grid. In this case, the last known good position of the particle is
        shifted to the boundary intersection point, while the new position
        is shifted to the reflected coordinates. It is anticipated this should
        quickly converge on a valid location, and an upper limit of 10
        iterations is imposed. If this limit is exceeded, a boundary error is 
        returned and the particle's position is not updated.
        """
        # 2D position vector for the reflected position
        cdef vector[DTYPE_FLOAT_t] x4_prime = vector[DTYPE_FLOAT_t](2, -999.)

        # 2D position and directon vectors used for locating lost particles
        cdef vector[DTYPE_FLOAT_t] x_test = vector[DTYPE_FLOAT_t](2, -999.)
        cdef vector[DTYPE_FLOAT_t] r_test = vector[DTYPE_FLOAT_t](2, -999.)
        
        # 2D directoion vector normal to the element side, pointing into the
        # element
        cdef vector[DTYPE_FLOAT_t] n = vector[DTYPE_FLOAT_t](2, -999.)
        
        # 2D direction vector pointing from xi to the new position
        cdef vector[DTYPE_FLOAT_t] d = vector[DTYPE_FLOAT_t](2, -999.)
        
        # 2D direction vector pointing from xi to x4', where x4' is the
        # reflected point that we ultimately trying to find
        cdef vector[DTYPE_FLOAT_t] r = vector[DTYPE_FLOAT_t](2, -999.)
        
        # Intermediate variable
        cdef DTYPE_FLOAT_t mult

        # Temporary particle objects
        cdef Particle particle_copy_a
        cdef Particle particle_copy_b

        # In domain flag
        cdef DTYPE_INT_t flag
        
        # Is found flag
        cdef bint found

        # Loop counters
        cdef DTYPE_INT_t counter_a, counter_b

        # The intersection point
        cdef Intersection intersection

        # Create copies of the two particles - these will be used to save
        # intermediate states
        particle_copy_a = particle_old[0]
        particle_copy_b = particle_new[0]

        counter_a = 0
        while counter_a < 10:
            # Compute coordinates for the side of the element the particle crossed
            # before exiting the domain
            intersection = data_reader.get_boundary_intersection(&particle_copy_a,
                                                                 &particle_copy_b)

            # Compute the direction vector pointing from the intersection point
            # to the position vector that lies outside of the model domain
            d[0] = particle_copy_b.get_x1() - intersection.xi
            d[1] = particle_copy_b.get_x2() - intersection.yi

            # Compute the normal to the element side that points back into the
            # element given the clockwise ordering of element vertices
            n[0] = intersection.y2 - intersection.y1
            n[1] = intersection.x1 - intersection.x2

            # Compute the reflection vector
            mult = 2.0 * inner_product(n, d) / inner_product(n, n)
            r[0] = d[0] - mult*n[0]
            x4_prime[0] = intersection.xi + r[0]

            r[1] = d[1] - mult*n[1]
            x4_prime[1] = intersection.yi + r[1]

            # Attempt to find the particle using a (cheap) local search
            # ---------------------------------------------------------
            particle_copy_b.set_x1(x4_prime[0])
            particle_copy_b.set_x2(x4_prime[1])
            flag = data_reader.find_host_using_local_search(&particle_copy_b)

            if flag == IN_DOMAIN:
                particle_new[0] = particle_copy_b
                return flag

            # Local search failed
            # -------------------
            
            # To locate the new particle's position, we first try to find a
            # location that sits safely in the model grid some way between
            # the intersection point and the new position. This will allow us to
            # use line crossings to definitely locate the particle. We don't use the
            # intersection point itself for this, as it could be flagged as a
            # line crossing, possibly affecting the result.
            #
            # Three attempts are made to find such an intermediate position, each
            # using global searching. The approach may fail, if the particle is
            # extremely close to the boundary. If this does happen, the search
            # is aborted and the particle is simply moved to the last known
            # host element's centroid.
            for i in xrange(2):
                r_test[i] = r[i]
            
            counter_b = 0
            while counter_b < 3:
                r_test[0] = r_test[0]/10.
                x_test[0] = intersection.xi + r_test[0]

                r_test[1] = r_test[1]/10.
                x_test[1] = intersection.yi + r_test[1]

                particle_copy_a.set_x1(x_test[0])
                particle_copy_a.set_x2(x_test[1])

                flag = data_reader.find_host_using_global_search(&particle_copy_a)

                if flag == IN_DOMAIN:
                    break

                counter_b += 1

            if flag != IN_DOMAIN:
                # Search failed. Reset the particle's position.
                data_reader.set_default_location(particle_new)
                return IN_DOMAIN

            # Attempt to locate the new point using a standard search
            # -------------------------------------------------------
            flag = data_reader.find_host(&particle_copy_a, &particle_copy_b)
            
            if flag == IN_DOMAIN:
                particle_new[0] = particle_copy_b
                return flag
            elif flag == OPEN_BDY_CROSSED:
                return flag
            
            counter_a += 1

        return BDY_ERROR


cdef class RestoringHorizGeographicBoundaryConditionCalculator(HorizBoundaryConditionCalculator):
    """ Restoring horizontal boundary condition calculator for geographic grids

    Restoring horizontal boundary condition calculators for geographic coordinate
    grids simply move the particle back to its last valid position.
    """
    cdef DTYPE_INT_t apply(self, DataReader data_reader, Particle *particle_old,
                           Particle *particle_new) except INT_ERR:
        cdef DTYPE_INT_t flag

        # Move the particle back to its last known valid position
        particle_new.set_x1(particle_old.get_x1())
        particle_new.set_x2(particle_old.get_x2())
        flag = data_reader.find_host(particle_old, particle_new)

        if flag == IN_DOMAIN:
            return flag

        return BDY_ERROR


cdef class RefHorizGeographicBoundaryConditionCalculator(HorizBoundaryConditionCalculator):
    """ Reflecting horizontal boundary condition calculator for geographic grids

    """
    cdef DTYPE_INT_t apply(self, DataReader data_reader, Particle *particle_old,
                           Particle *particle_new) except INT_ERR:
        """Apply reflecting boundary conditions

        The algorithm computes reflection vectors for particles given
        information that describes their last known good position (which was
        inside the domain) and their new position which has been flagged as
        lying outside of the domain. It does this by first calculating where
        the particle pathline intersected the model boundary. The process is
        iterative, allowing for situations in which the reflected point
        is still outside the domain - this can happen in corners of the model
        grid. In this case, the last known good position of the particle is
        shifted to the boundary intersection point, while the new position
        is shifted to the reflected coordinates. It is anticipated this should
        quickly converge on a valid location, and an upper limit of 10
        iterations is imposed. If this limit is exceeded, a boundary error is
        returned and the particle's position is not updated.

        The process is very similar to that in Cartesian coordinates. Distinct
        steps include converting to cartesian coordinates, then rotating the
        cartesian axes so that the positive z-axis forms an outward normal
        through the intersection point, while the x- and y- axes are locally
        aligned with lines of constant longitude and latitude respectively.
        All points are then projected onto the plane that lies tangential to
        the surface of the sphere at the intersection point. The reflection
        vector is then computed in the rotated coordinate system. Finally,
        the coordinates of the reflected point are converted back to geographic
        coordinates and rotated back to the original frame of reference. The
        whole process is iterative, as described above.
        """
        # Locations in Cartesian coordinates
        cdef vector[DTYPE_FLOAT_t] pi = vector[DTYPE_FLOAT_t](3, -999.)
        cdef vector[DTYPE_FLOAT_t] pa = vector[DTYPE_FLOAT_t](3, -999.)
        cdef vector[DTYPE_FLOAT_t] pb = vector[DTYPE_FLOAT_t](3, -999.)
        cdef vector[DTYPE_FLOAT_t] p1 = vector[DTYPE_FLOAT_t](3, -999.)
        cdef vector[DTYPE_FLOAT_t] p2 = vector[DTYPE_FLOAT_t](3, -999.)

        # Position vectors for the reflected position
        cdef vector[DTYPE_FLOAT_t] x4_prime = vector[DTYPE_FLOAT_t](3, -999.)
        cdef vector[DTYPE_FLOAT_t] x4_prime_geog = vector[DTYPE_FLOAT_t](2, -999.)

        # 2D position and direction vectors used for locating lost particles
        cdef vector[DTYPE_FLOAT_t] r_test = vector[DTYPE_FLOAT_t](3, -999.)
        cdef vector[DTYPE_FLOAT_t] x_test = vector[DTYPE_FLOAT_t](3, -999.)
        cdef vector[DTYPE_FLOAT_t] x_test_geog = vector[DTYPE_FLOAT_t](2, -999.)

        # 2D directoion vector normal to the element side, pointing into the
        # element
        cdef vector[DTYPE_FLOAT_t] n = vector[DTYPE_FLOAT_t](2, -999.)

        # 2D direction vector pointing from xi to the new position
        cdef vector[DTYPE_FLOAT_t] d = vector[DTYPE_FLOAT_t](2, -999.)

        # 2D direction vector pointing from xi to x4', where x4' is the
        # reflected point that we ultimately trying to find
        cdef vector[DTYPE_FLOAT_t] r = vector[DTYPE_FLOAT_t](2, -999.)

        # Intermediate variable
        cdef DTYPE_FLOAT_t mult

        # Temporary particle objects
        cdef Particle particle_copy_a
        cdef Particle particle_copy_b

        # In domain flag
        cdef DTYPE_INT_t flag

        # Is found flag
        cdef bint found

        # Loop counters
        cdef DTYPE_INT_t counter_a, counter_b

        # The intersection point
        cdef Intersection intersection

        # Create copies of the two particles - these will be used to save
        # intermediate states
        particle_copy_a = particle_old[0]
        particle_copy_b = particle_new[0]

        counter_a = 0
        while counter_a < 10:
            # Compute coordinates for the side of the element the particle crossed
            # before exiting the domain
            intersection = data_reader.get_boundary_intersection(&particle_copy_a,
                                                                 &particle_copy_b)

            # Convert to cartesian coordinates
            pi = geographic_to_cartesian_coords(intersection.xi, intersection.yi, 1.0)
            pa = geographic_to_cartesian_coords(particle_copy_a.get_x1(), particle_copy_a.get_x2(), 1.0)
            pb = geographic_to_cartesian_coords(particle_copy_b.get_x1(), particle_copy_b.get_x2(), 1.0)
            p1 = geographic_to_cartesian_coords(intersection.x1, intersection.y1, 1.0)
            p2 = geographic_to_cartesian_coords(intersection.x2, intersection.y2, 1.0)

            # Rotate axes to get the desired orientation as described above
            pi = rotate_axes(pi, intersection.xi, intersection.yi)
            pa = rotate_axes(pa, intersection.xi, intersection.yi)
            pb = rotate_axes(pb, intersection.xi, intersection.yi)
            p1 = rotate_axes(p1, intersection.xi, intersection.yi)
            p2 = rotate_axes(p2, intersection.xi, intersection.yi)

            # Compute the direction vector pointing from the intersection point
            # to the position vector that lies outside of the model domain
            d[0] = pb[0] - pi[0]
            d[1] = pb[1] - pi[1]

            # Compute the normal to the element side that points back into the
            # element given the clockwise ordering of element vertices
            n[0] = p2[1] - p1[1]
            n[1] = p1[0] - p2[0]

            # Compute the reflection vector
            mult = 2.0 * inner_product(n, d) / inner_product(n, n)
            r[0] = d[0] - mult*n[0]
            x4_prime[0] = pi[0] + r[0]

            r[1] = d[1] - mult*n[1]
            x4_prime[1] = pi[1] + r[1]

            # Set z to 1.0
            x4_prime[2] = 1.0

            # Attempt to find the particle using a (cheap) local search
            # ---------------------------------------------------------
            x4_prime = reverse_rotate_axes(x4_prime, intersection.xi, intersection.yi)
            x4_prime_geog = cartesian_to_geographic_coords(x4_prime)
            particle_copy_b.set_x1(x4_prime_geog[0])
            particle_copy_b.set_x2(x4_prime_geog[1])
            flag = data_reader.find_host_using_local_search(&particle_copy_b)

            if flag == IN_DOMAIN:
                particle_new[0] = particle_copy_b
                return flag

            # Local search failed
            # -------------------

            # To locate the new particle's position, we first try to find a
            # location that sits safely in the model grid some way between
            # the intersection point and the new position. This will allow us to
            # use line crossings to definitely locate the particle. We don't use the
            # intersection point itself for this, as it could be flagged as a
            # line crossing, possibly affecting the result.
            #
            # Three attempts are made to find such an intermediate position, each
            # using global searching. The approach may fail, if the particle is
            # extremely close to the boundary. If this does happen, the search
            # is aborted and the particle is simply moved to the last known
            # host element's centroid.
            for i in xrange(2):
                r_test[i] = r[i]
            r_test[2] = 1.0

            counter_b = 0
            while counter_b < 3:
                r_test[0] = r_test[0]/10.
                x_test[0] = pi[0] + r_test[0]

                r_test[1] = r_test[1]/10.
                x_test[1] = pi[1] + r_test[1]

                x_test = reverse_rotate_axes(x_test, intersection.xi, intersection.yi)
                x_test_geog = cartesian_to_geographic_coords(x_test)

                particle_copy_a.set_x1(x_test[0])
                particle_copy_a.set_x2(x_test[1])

                flag = data_reader.find_host_using_global_search(&particle_copy_a)

                if flag == IN_DOMAIN:
                    break

                counter_b += 1

            if flag != IN_DOMAIN:
                # Search failed. Reset the particle's position.
                data_reader.set_default_location(particle_new)
                return IN_DOMAIN

            # Attempt to locate the new point using a standard search
            # -------------------------------------------------------
            flag = data_reader.find_host(&particle_copy_a, &particle_copy_b)

            if flag == IN_DOMAIN:
                particle_new[0] = particle_copy_b
                return flag
            elif flag == OPEN_BDY_CROSSED:
                return flag

            counter_a += 1

        return BDY_ERROR


cdef class VertBoundaryConditionCalculator:

    def apply_wrapper(self, DataReader data_reader,
                      DTYPE_FLOAT_t time,
                      ParticleSmartPtr particle):
        """ Python friendly wrapper for apply()

        The apply() method must be implemented in the derived class.
        It should apply the selected vertical boundary condition in response
        to a particle crossing a vertical boundary, and update the position of
        `particle_new`.

        Parameters
        ----------
        data_reader : pylag.data_reader.DataReader
            A concrete PyLag data reader which inherits from the base class
            `pylag.data_reader.DataReader`.

        time : float
            The time the crossing occurred.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle after it has crossed the vertical boundary. The particle
            should lie inside the model domain.

        Returns
        -------
         : int
            Flag signifying whether or not the application was successful.
        """
        return self.apply(data_reader, time, particle.get_ptr())

    cdef DTYPE_INT_t apply(self, DataReader data_reader, DTYPE_FLOAT_t time, 
                           Particle *particle) except INT_ERR:
        raise NotImplementedError


cdef class RefVertBoundaryConditionCalculator(VertBoundaryConditionCalculator):
    """ Reflecting vertical boundary condition calculator

    The calculator reflects the particle back into the domain. It may apply
    multiple reflections if required. The particle's position is updated in
    place.
    """

    cdef DTYPE_INT_t apply(self, DataReader data_reader, DTYPE_FLOAT_t time, 
                           Particle *particle) except INT_ERR:
        cdef DTYPE_FLOAT_t zmin
        cdef DTYPE_FLOAT_t zmax
        cdef DTYPE_FLOAT_t x3
        cdef DTYPE_INT_t flag

        zmin = data_reader.get_zmin(time, particle)
        zmax = data_reader.get_zmax(time, particle)
        x3 = particle.get_x3()

        if zmin < zmax:
            while x3 < zmin or x3 > zmax:
                if x3 < zmin:
                    x3 = zmin + zmin - x3
                elif x3 > zmax:
                    x3 = zmax + zmax - x3
        else:
            # Cell is dry. Place the particle on the sea floor.
            x3 = zmin

        particle.set_x3(x3)

        flag = data_reader.set_vertical_grid_vars(time, particle)

        if flag != IN_DOMAIN:
            # This could happen if the grid search was performed using sigma
            # coords and the computed sigma coordinate lies outside of the
            # domain, even though x3 lies within the domain. To correct for
            # this, adjust x3 by epsilon scaled by the column depth.
            if abs(particle.get_x3() - zmax) < abs(particle.get_x3() - zmin):
                particle.set_x3(particle.get_x3() - EPSILON * (zmax - zmin))
            else:
                particle.set_x3(particle.get_x3() + EPSILON * (zmax - zmin))
 
            flag = data_reader.set_vertical_grid_vars(time, particle)

        return flag


# Factory methods
# ---------------

def get_horiz_boundary_condition_calculator(config):
    """ Factory method for horizontal boundary condition calculators

    Parameters
    ----------
    config : ConfigParser
        PyLag configuraton object

    Returns
    -------
     : HorizontalBoundaryConditionCalculator
         A horizontal boundary condition calculator
    """
    try:
        boundary_condition = config.get("BOUNDARY_CONDITIONS", "horiz_bound_cond")
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        return None
    else:
        coordinate_system = config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system")

        if coordinate_system not in ['cartesian', 'geographic']:
            raise ValueError('Unsupported coordinate system.')

        if boundary_condition == "restoring":
            coordinate_system = config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system")
            if coordinate_system == "cartesian":
                return RestoringHorizCartesianBoundaryConditionCalculator()
            elif coordinate_system == "geographic":
                return RestoringHorizGeographicBoundaryConditionCalculator()

        elif boundary_condition == "reflecting":
            coordinate_system = config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system")
            if coordinate_system == "cartesian":
                return RefHorizCartesianBoundaryConditionCalculator()
            elif coordinate_system == "geographic":
                return RefHorizGeographicBoundaryConditionCalculator()

        elif boundary_condition == "None":
            return None
        else:
            raise ValueError('Unsupported horizontal boundary condition.')


def get_vert_boundary_condition_calculator(config):
    """ Factory method for vertical boundary condition calculators

    Parameters
    ----------
    config : ConfigParser
        PyLag configuraton object

    Returns
    -------
     : VerticalBoundaryConditionCalculator
         A vertical boundary condition calculator
    """
    try:
        boundary_condition = config.get("BOUNDARY_CONDITIONS", "vert_bound_cond")
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        return None
    else:
        if boundary_condition == "reflecting":
            return RefVertBoundaryConditionCalculator()
        elif boundary_condition == "None":
            return None
        else:
            raise ValueError('Unsupported vertical boundary condtion.')

