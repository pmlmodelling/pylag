include "constants.pxi"

cdef class DataReader:
    cpdef setup_data_access(self, start_datetime, end_datetime):
        NotImplementedError

    cpdef read_data(self, DTYPE_FLOAT_t time):
        NotImplementedError

    cpdef find_host(self, DTYPE_FLOAT_t xpos_old, DTYPE_FLOAT_t ypos_old,
            DTYPE_FLOAT_t xpos_new, DTYPE_FLOAT_t ypos_new, DTYPE_INT_t guess):
        NotImplementedError

    cpdef find_host_using_global_search(self, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos):
        NotImplementedError

    cpdef find_host_using_local_search(self, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t first_guess):
        NotImplementedError

    cpdef get_boundary_intersection(self, DTYPE_FLOAT_t xpos_old,
            DTYPE_FLOAT_t ypos_old, DTYPE_FLOAT_t xpos_new,
            DTYPE_FLOAT_t ypos_new, DTYPE_INT_t last_host):
        NotImplementedError

    cdef set_local_coordinates(self, Particle *particle):
        NotImplementedError

    cdef set_vertical_grid_vars(self, DTYPE_FLOAT_t time, Particle *particle):
        NotImplementedError

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle):
        NotImplementedError

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle):
        NotImplementedError

    cdef get_velocity(self, DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t vel[3]):
        NotImplementedError

    cdef get_horizontal_velocity(self, DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t vel[2]):
        NotImplementedError
    
    cdef get_vertical_velocity(self, DTYPE_FLOAT_t time, Particle *particle):
        NotImplementedError

    cdef get_horizontal_eddy_viscosity(self, DTYPE_FLOAT_t time,
            Particle *particle):
        NotImplementedError

    cdef get_horizontal_eddy_viscosity_derivative(self, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t Ah_prime[2]):
        NotImplementedError

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            Particle *particle) except FLOAT_ERR:
        NotImplementedError
    
    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self, 
            DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        NotImplementedError
    
    cpdef DTYPE_INT_t is_wet(self, DTYPE_FLOAT_t time, DTYPE_INT_t host) except INT_ERR:
        NotImplementedError
