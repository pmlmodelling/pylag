cdef class DataReader:
    cpdef setup_data_access(self, start_datetime, end_datetime):
        pass

    cpdef read_data(self, DTYPE_FLOAT_t time):
        pass

    cpdef find_host(self, DTYPE_FLOAT_t xpos_old, DTYPE_FLOAT_t ypos_old,
            DTYPE_FLOAT_t xpos_new, DTYPE_FLOAT_t ypos_new, DTYPE_INT_t guess):
        pass

    cpdef get_boundary_intersection(self, DTYPE_FLOAT_t xpos_old,
            DTYPE_FLOAT_t ypos_old, DTYPE_FLOAT_t xpos_new,
            DTYPE_FLOAT_t ypos_new, DTYPE_INT_t last_host):
        pass

    cpdef find_zlayer(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
        DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
        DTYPE_INT_t guess):
        pass

    cpdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t host):
        pass

    cpdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t host):
        pass

    cdef get_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_INT_t zlayer, DTYPE_FLOAT_t vel[3]):
        pass

    cdef get_horizontal_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_INT_t zlayer, DTYPE_FLOAT_t vel[2]):
        pass
    
    cdef get_vertical_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_INT_t zlayer):
        pass

    cpdef get_horizontal_eddy_diffusivity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_INT_t zlayer):
        pass

    cpdef get_horizontal_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t time,
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos,
            DTYPE_INT_t host, DTYPE_INT_t zlayer):
        pass

    cpdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_INT_t zlayer):
        pass
    
    cpdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t time,
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos,
            DTYPE_INT_t host, DTYPE_INT_t zlayer):
        pass
