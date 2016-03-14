# Data types used for constructing C data structures
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef class DataReader:
    cpdef find_host(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_INT_t guess)

    cpdef update_time_dependent_vars(self, DTYPE_FLOAT_t time) 

    cpdef get_bathymetry(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, 
            DTYPE_INT_t host)

    cpdef get_sea_sur_elev(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t host)

    cdef get_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host, 
            DTYPE_FLOAT_t vel[3])

    cdef get_horizontal_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host, 
            DTYPE_FLOAT_t vel[2])

    cdef get_vertical_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host)

    cdef get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host)

    cdef get_vertical_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t time,
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos,
            DTYPE_INT_t host)
