include "constants.pxi"

from libcpp.string cimport string
from libcpp.vector cimport vector

# Data types used for constructing C data structures
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from pylag.particle cimport Particle


cdef class Grid:

    cdef DTYPE_INT_t find_host_using_global_search(self,
                                                   Particle *particle) except INT_ERR

    cdef DTYPE_INT_t find_host_using_local_search(self,
                                                  Particle *particle) except INT_ERR

    cdef DTYPE_INT_t find_host_using_particle_tracing(self, Particle *particle_old,
                                                      Particle *particle_new) except INT_ERR

    cdef get_boundary_intersection(self,
                                   Particle *particle_old,
                                   Particle *particle_new,
                                   DTYPE_FLOAT_t elem_side[2],
                                   DTYPE_FLOAT_t particle_pathline[2],
                                   DTYPE_FLOAT_t intersection[2])

    cdef set_default_location(self, Particle *particle)

    cdef set_local_coordinates(self, Particle *particle)

    cpdef vector[DTYPE_FLOAT_t] get_phi(self, const DTYPE_FLOAT_t &x1, const DTYPE_FLOAT_t &x2, const DTYPE_INT_t &host)

    cdef void get_grad_phi(self, DTYPE_INT_t host,
                           DTYPE_FLOAT_t dphi_dx[3],
                           DTYPE_FLOAT_t dphi_dy[3]) except *

    cdef DTYPE_FLOAT_t interpolate_in_space(self, DTYPE_FLOAT_t[::1] var_arr, Particle *particle) except FLOAT_ERR

    cdef DTYPE_FLOAT_t interpolate_in_time_and_space_2D(self, DTYPE_FLOAT_t[::1] var_last_arr,
                                                        DTYPE_FLOAT_t[::1] var_next_arr,
                                                        DTYPE_FLOAT_t time_fraction, Particle *particle) except FLOAT_ERR

    cdef DTYPE_FLOAT_t interpolate_in_time_and_space(self, DTYPE_FLOAT_t[:, ::1] var_last_arr,
                                                     DTYPE_FLOAT_t[:, ::1] var_next_arr, DTYPE_INT_t k,
                                                     DTYPE_FLOAT_t time_fraction, Particle *particle) except FLOAT_ERR

    cdef void interpolate_grad_in_time_and_space(self, const DTYPE_FLOAT_t[:, ::1] &var_last_arr,
                                                 const DTYPE_FLOAT_t[:, ::1] &var_next_arr, DTYPE_INT_t k,
                                                 DTYPE_FLOAT_t time_fraction, Particle *particle,
                                                 DTYPE_FLOAT_t var_prime[2]) except *

    cdef DTYPE_FLOAT_t shepard_interpolation(self, const DTYPE_FLOAT_t &x,
            const DTYPE_FLOAT_t &y, const DTYPE_FLOAT_t xpts[4], const DTYPE_FLOAT_t ypts[4],
            const DTYPE_FLOAT_t vals[4], const DTYPE_INT_t valid_points[4]) except FLOAT_ERR