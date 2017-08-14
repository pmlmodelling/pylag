include "constants.pxi"

# Data types used for constructing C data structures
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle cimport Particle

cdef class DataReader:
    cpdef setup_data_access(self, start_datetime, end_datetime)

    cpdef read_data(self, DTYPE_FLOAT_t time) 

    cpdef find_host(self, DTYPE_FLOAT_t xpos_old, DTYPE_FLOAT_t ypos_old,
        DTYPE_FLOAT_t xpos_new, DTYPE_FLOAT_t ypos_new, DTYPE_INT_t guess)

    cpdef find_host_using_global_search(self, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos)

    cpdef find_host_using_local_search(self, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t first_guess)

    cpdef get_boundary_intersection(self, DTYPE_FLOAT_t xpos_old,
        DTYPE_FLOAT_t ypos_old, DTYPE_FLOAT_t xpos_new, DTYPE_FLOAT_t ypos_new,
        DTYPE_INT_t last_host)

    cdef set_local_coordinates(self, Particle *particle)

    cdef set_vertical_grid_vars(self, DTYPE_FLOAT_t time, Particle *particle)

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle)

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle)

    cdef get_velocity(self, DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t vel[3])

    cdef get_horizontal_velocity(self, DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t vel[2])

    cdef get_vertical_velocity(self, DTYPE_FLOAT_t time, Particle *particle)

    cdef get_horizontal_eddy_viscosity(self, DTYPE_FLOAT_t time,
            Particle *particle)

    cdef get_horizontal_eddy_viscosity_derivative(self, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t Ah_prime[2])

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            Particle* particle) except FLOAT_ERR

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self,
            DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR
