include "constants.pxi"

import logging

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

# PyLag cimports
from pylag.particle_cpp_wrapper cimport ParticleSmartPtr
from pylag.math cimport inner_product
from pylag.math cimport Intersection

cdef class HorizBoundaryConditionCalculator:

    def apply_wrapper(self, DataReader data_reader,
                      ParticleSmartPtr particle_old,
                      ParticleSmartPtr particle_new):
        """ Python friendly wrapper for apply()
        
        """

        return self.apply(data_reader, particle_old.get_ptr(), particle_new.get_ptr())

    cdef DTYPE_INT_t apply(self, DataReader data_reader, Particle *particle_old,
                           Particle *particle_new) except INT_ERR:
        raise NotImplementedError

cdef class RefHorizBoundaryConditionCalculator(HorizBoundaryConditionCalculator):

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
        cdef DTYPE_FLOAT_t x4_prime[2]
        
        # 2D position and directon vectors used for locating lost particles
        cdef DTYPE_FLOAT_t x_test[2]
        cdef DTYPE_FLOAT_t r_test[2]
        
        # 2D directoion vector normal to the element side, pointing into the
        # element
        cdef DTYPE_FLOAT_t n[2]     
        
        # 2D direction vector pointing from xi to the new position
        cdef DTYPE_FLOAT_t d[2]
        
        # 2D direction vector pointing from xi to x4', where x4' is the
        # reflected point that we ultimately trying to find
        cdef DTYPE_FLOAT_t r[2]
        
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
            flag = data_reader.find_host_using_local_search(&particle_copy_b,
                                                            particle_copy_b.get_host_horizontal_elem())

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


cdef class VertBoundaryConditionCalculator:

    def apply_wrapper(self, DataReader data_reader,
                      DTYPE_FLOAT_t time,
                      ParticleSmartPtr particle):
        """ Python friendly wrapper for apply()
        
        """

        return self.apply(data_reader, time, particle.get_ptr())

    cdef DTYPE_INT_t apply(self, DataReader data_reader, DTYPE_FLOAT_t time, 
                           Particle *particle) except INT_ERR:
        raise NotImplementedError

cdef class RefVertBoundaryConditionCalculator(VertBoundaryConditionCalculator):
    """ Apply reflecting vertical boundary condition in cartesian coords """

    cdef DTYPE_INT_t apply(self, DataReader data_reader, DTYPE_FLOAT_t time, 
                           Particle *particle) except INT_ERR:
        cdef DTYPE_FLOAT_t zmin
        cdef DTYPE_FLOAT_t zmax
        cdef DTYPE_FLOAT_t x3
        cdef DTYPE_INT_t flag

        zmin = data_reader.get_zmin(time, particle)
        zmax = data_reader.get_zmax(time, particle)
        x3 = particle.get_x3()

        while x3 < zmin or x3 > zmax:
            if x3 < zmin:
                x3 = zmin + zmin - x3
            elif x3 > zmax:
                x3 = zmax + zmax - x3

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
    try:
        boundary_condition = config.get("BOUNDARY_CONDITIONS", "horiz_bound_cond")
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        return None
    else:
        if boundary_condition == "reflecting":
            return RefHorizBoundaryConditionCalculator()
        elif boundary_condition == "None":
            return None
        else:
            raise ValueError('Unsupported horizontal boundary condtion.')

def get_vert_boundary_condition_calculator(config):
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

