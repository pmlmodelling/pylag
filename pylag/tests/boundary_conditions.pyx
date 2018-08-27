include "constants.pxi"

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t
from pylag.data_reader cimport DataReader
from pylag.particle cimport Particle

cdef class TestHorizBCDataReader(DataReader):
    
    cdef DTYPE_INT_t find_host(self, Particle *particle_old,
                               Particle *particle_new) except INT_ERR:
        return IN_DOMAIN

    cdef DTYPE_INT_t find_host_using_local_search(self,  Particle *particle,
                                       DTYPE_INT_t first_guess) except INT_ERR:
        return IN_DOMAIN

    cdef DTYPE_INT_t find_host_using_global_search(self,  Particle *particle) except INT_ERR:
        return IN_DOMAIN

    cpdef get_boundary_intersection(self, DTYPE_FLOAT_t xpos_old,
                                    DTYPE_FLOAT_t ypos_old,
                                    DTYPE_FLOAT_t xpos_new,
                                    DTYPE_FLOAT_t ypos_new,
                                    DTYPE_INT_t last_host):
        """ Get boundary intersection
        
        Test function assumes the pathline intersected a line with coordinates
        (-1,-1) and (1,1) at the point (0,0). All function arguments are
        ignored.
        
        Consistent choices for testing are:
        xpos_old = 0.0
        ypos_old = -1.0
        xpos_new = 0.0
        ypos_new = 1.0
        host = 0
        """

        return -1.0, -1.0, 1.0, 1.0, 0.0, 0.0
