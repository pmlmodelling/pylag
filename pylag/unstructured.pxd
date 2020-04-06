include "constants.pxi"

from libcpp.vector cimport vector

# Data types used for constructing C data structures
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from pylag.particle cimport Particle
from math cimport Intersection

cdef class UnstructuredGrid:

    # Configurtion object
    cdef object config

    # Grid dimensions
    cdef DTYPE_INT_t n_elems, n_nodes

    # Element connectivity
    cdef DTYPE_INT_t[:,:] nv

    # Element adjacency
    cdef DTYPE_INT_t[:,:] nbe

    # Nodal coordinates
    cdef DTYPE_FLOAT_t[:] x
    cdef DTYPE_FLOAT_t[:] y

    # Element centre coordinates
    cdef DTYPE_FLOAT_t[:] xc
    cdef DTYPE_FLOAT_t[:] yc

    cdef DTYPE_INT_t find_host_using_global_search(self,
                                                   Particle *particle) except INT_ERR

    cdef DTYPE_INT_t find_host_using_local_search(self,
                                                  Particle *particle,
                                                  DTYPE_INT_t first_guess) except INT_ERR

    cdef DTYPE_INT_t find_host_using_particle_tracing(self, Particle *particle_old,
                                                      Particle *particle_new) except INT_ERR

    cdef Intersection get_boundary_intersection(self, Particle *particle_old,
                                                Particle *particle_new)

    cdef set_default_location(self, Particle *particle)

    cdef set_local_coordinates(self, Particle *particle)

    cdef void get_phi(self, DTYPE_FLOAT_t x1, DTYPE_FLOAT_t x2,
                      DTYPE_INT_t host, vector[DTYPE_FLOAT_t] &phi) except *

    cdef void get_grad_phi(self, DTYPE_INT_t host,
                           vector[DTYPE_FLOAT_t] &dphi_dx,
                           vector[DTYPE_FLOAT_t] &dphi_dy) except *
