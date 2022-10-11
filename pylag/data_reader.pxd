include "constants.pxi"

from libcpp.vector cimport vector

# Data types used for constructing C data structures
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT

# PyLag cimports
from pylag.particle cimport Particle


cdef class DataReader:
    cpdef setup_data_access(self, start_datetime, end_datetime)

    cpdef read_data(self, DTYPE_FLOAT_t time) 

    cpdef get_grid_names(self)

    cdef DTYPE_INT_t find_host(self, Particle *particle_old,
                               Particle *particle_new) except INT_ERR

    cdef DTYPE_INT_t find_host_using_global_search(self,
                                                   Particle *particle) except INT_ERR

    cdef DTYPE_INT_t find_host_using_local_search(self,
                                                  Particle *particle) except INT_ERR

    cdef get_boundary_intersection(self,
                                   Particle *particle_old,
                                   Particle *particle_new,
                                   DTYPE_FLOAT_t elem_side[2],
                                   DTYPE_FLOAT_t particle_pathline[2],
                                   DTYPE_FLOAT_t intersection[2])

    cdef set_default_location(self, Particle *particle)

    cdef set_local_coordinates(self, Particle *particle)

    cdef DTYPE_INT_t set_vertical_grid_vars(self, DTYPE_FLOAT_t time,
                                            Particle *particle) except INT_ERR

    cpdef DTYPE_FLOAT_t get_xmin(self) except FLOAT_ERR

    cpdef DTYPE_FLOAT_t get_ymin(self) except FLOAT_ERR

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR

    cdef void get_velocity(self, DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t vel[3]) except +

    cdef void get_horizontal_velocity(self, DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t vel[2]) except +

    cdef DTYPE_FLOAT_t get_vertical_velocity(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR

    cdef DTYPE_FLOAT_t get_horizontal_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            Particle *particle) except FLOAT_ERR

    cdef void get_horizontal_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t Ah_prime[2]) except +

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            Particle* particle) except FLOAT_ERR

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self,
            DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR

    cdef DTYPE_INT_t is_wet(self, DTYPE_FLOAT_t time, Particle *particle) except INT_ERR

    cdef DTYPE_FLOAT_t get_environmental_variable(self, var_name,
            DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR

    cdef void get_ten_meter_wind_velocity(self, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t wind_vel[2]) except +

    cdef void get_surface_stokes_drift_velocity(self, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t stokes_drift[2]) except +
