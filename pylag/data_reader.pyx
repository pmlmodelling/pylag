# Data types used for constructing C data structures
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef class DataReader:
    cpdef update_time_dependent_vars(self, DTYPE_FLOAT_t time):
        pass
    
    cpdef get_bathymetry(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, 
            DTYPE_INT_t host):
        pass
    
    cpdef get_sea_sur_elev(self, DTYPE_FLOAT_t time_fraction, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t host):
        pass

    cdef get_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host, 
            DTYPE_FLOAT_t[:] vel):
        pass
    
    cpdef find_host(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_INT_t guess):
        pass
    
    cpdef sigma_to_cartesian_coords(self, DTYPE_FLOAT_t sigma, DTYPE_FLOAT_t h,
            DTYPE_FLOAT_t zeta):
        pass

    cpdef cartesian_to_sigma_coords(self, DTYPE_FLOAT_t z, DTYPE_FLOAT_t h,
            DTYPE_FLOAT_t zeta):
        pass