include "constants.pxi"

import logging

# PyLag cimports
from pylag.particle cimport ParticleSmartPtr
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
        # 2D position vectors for the end points of the element's side
        cdef DTYPE_FLOAT_t x1[2]
        cdef DTYPE_FLOAT_t x2[2]
        
        # 2D position vectors for the particle's previous and new position
        cdef DTYPE_FLOAT_t x3[2]
        cdef DTYPE_FLOAT_t x4[2]
        
        # 2D position vector for the intersection point
        cdef DTYPE_FLOAT_t xi[2]
        
        # 2D position vector for the reflected position
        cdef DTYPE_FLOAT_t x4_prime[2]
        
        # 2D position and directon vectors used for locating lost particles
        cdef DTYPE_FLOAT_t x_test[2]
        cdef DTYPE_FLOAT_t r_test[2]
        
        # 2D directoion vector normal to the element side, pointing into the
        # element
        cdef DTYPE_FLOAT_t n[2]     
        
        # 2D direction vector pointing from xi to x4
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

        # Loop counter
        cdef DTYPE_INT_t counter

        # The intersection point
        cdef Intersection intersection

        # Create copies of the two particles - these will be used to save
        # intermediate states
        particle_copy_a = particle_old[0]
        particle_copy_b = particle_new[0]

        counter = 0
        while counter < 10:
            # Construct arrays to hold the coordinates of the particle's previous
            # position vector and its new position vector that lies outside of the
            # model domain
            x3[0] = particle_copy_a.xpos
            x3[1] = particle_copy_a.ypos
            x4[0] = particle_copy_b.xpos
            x4[1] = particle_copy_b.ypos

            # Compute coordinates for the side of the element the particle crossed
            # before exiting the domain
            intersection = data_reader.get_boundary_intersection(&particle_copy_a,
                                                                 &particle_copy_b)

            # Compute the direction vector pointing from the intersection point
            # to the position vector that lies outside of the model domain
            d[0] = x4[0] - intersection.xi
            d[1] = x4[1] - intersection.yi

            # Compute the normal to the element side that points back into the
            # element given the clockwise ordering of element vertices
            n[0] = intersection.y2 - intersection.y1
            n[1] = intersection.x1 - intersection.x2

            # Compute the reflection vector
            mult = 2.0 * inner_product(n, d) / inner_product(n, n)
            for i in xrange(2):
                r[i] = d[i] - mult*n[i]
                x4_prime[i] = xi[i] + r[i]

            # Attempt to find the particle using a local search
            # -------------------------------------------------
            particle_copy_b.xpos = x4_prime[0]
            particle_copy_b.ypos = x4_prime[1]
            flag = data_reader.find_host_using_local_search(&particle_copy_b,
                                                            particle_copy_b.host_horizontal_elem)

            if flag == IN_DOMAIN:
                particle_new[0] = particle_copy_b
                return flag

            # Local search failed
            # -------------------
            
            # To locate the new particle's position, first find a location that
            # sits safely in the model grid some way between the intersection
            # point and the new position. This will allow us to use line
            # crossings to definitely locate the particle. We don't use the
            # intersection point itself for this, for fear that it will be
            # flagged as a line crossing. To guarantee the search, we resort
            # to global searching.
            for i in xrange(2):
                r_test[i] = r[i]
            
            found = False
            while found == False:
                for i in xrange(2):
                    r_test[i] = r_test[i]/10.
                    x_test[i] = xi[i] + r_test[i]

                particle_copy_a.xpos = x_test[0]
                particle_copy_a.ypos = x_test[1]

                flag = data_reader.find_host_using_global_search(&particle_copy_a)

                if flag == IN_DOMAIN:
                    found = True

            # Attempt to locate the new point using robust searching
            # ------------------------------------------------------
            flag = data_reader.find_host(&particle_copy_a, &particle_copy_b)
            
            if flag == IN_DOMAIN:
                particle_new[0] = particle_copy_b
                return flag
            elif flag == OPEN_BDY_CROSSED:
                return flag
            
            counter += 1
        
        return BDY_ERROR


cdef class VertBoundaryConditionCalculator:

     cpdef DTYPE_FLOAT_t apply(self, DTYPE_FLOAT_t zpos, DTYPE_FLOAT_t zmin,
             DTYPE_FLOAT_t zmax):
        pass

cdef class RefVertBoundaryConditionCalculator(VertBoundaryConditionCalculator):

     cpdef DTYPE_FLOAT_t apply(self, DTYPE_FLOAT_t zpos, DTYPE_FLOAT_t zmin,
             DTYPE_FLOAT_t zmax):
        """Apply reflecting boundary conditions
        
        """
        while zpos < zmin or zpos > zmax:
            if zpos < zmin:
                zpos = zmin + zmin - zpos
            elif zpos > zmax:
                zpos = zmax + zmax - zpos

        return zpos

# Factory methods
# ---------------

def get_horiz_boundary_condition_calculator(config):
    if config.get("BOUNDARY_CONDITIONS", "horiz_bound_cond") == "reflecting":
        return RefHorizBoundaryConditionCalculator()
    elif config.get("BOUNDARY_CONDITIONS", "horiz_bound_cond") == "None":
        return None
    else:
        raise ValueError('Unsupported horizontal boundary condtion.')

def get_vert_boundary_condition_calculator(config):
    if config.get("BOUNDARY_CONDITIONS", "vert_bound_cond") == "reflecting":
        return RefVertBoundaryConditionCalculator()
    elif config.get("BOUNDARY_CONDITIONS", "vert_bound_cond") == "None":
        return None
    else:
        raise ValueError('Unsupported vertical boundary condtion.')
