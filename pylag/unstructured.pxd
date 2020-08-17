include "constants.pxi"

from libcpp.string cimport string
from libcpp.vector cimport vector

# Data types used for constructing C data structures
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from pylag.particle cimport Particle
from pylag.math cimport Intersection

cdef class Grid:



    cdef DTYPE_INT_t find_host_using_global_search(self,
                                                   Particle *particle) except INT_ERR

    cdef DTYPE_INT_t find_host_using_local_search(self,
                                                  Particle *particle) except INT_ERR

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

    cdef DTYPE_FLOAT_t interpolate_in_space(self, DTYPE_FLOAT_t[:] var_arr, Particle *particle) except FLOAT_ERR

    cdef DTYPE_FLOAT_t interpolate_in_time_and_space(self, DTYPE_FLOAT_t[:] var_last_arr, DTYPE_FLOAT_t[:] var_next_arr,
                                                     DTYPE_FLOAT_t time_fraction, Particle *particle) except FLOAT_ERR

    cdef void interpolate_grad_in_time_and_space(self, DTYPE_FLOAT_t[:] var_last_arr, DTYPE_FLOAT_t[:] var_next_arr,
                                                 DTYPE_FLOAT_t time_fraction, Particle *particle,
                                                 DTYPE_FLOAT_t var_prime[2]) except *
