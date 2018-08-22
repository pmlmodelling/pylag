include "constants.pxi"

from pylag.particle cimport ParticleSmartPtr

cdef class DataReader:
    cpdef setup_data_access(self, start_datetime, end_datetime):
        raise NotImplementedError

    cpdef read_data(self, DTYPE_FLOAT_t time):
        raise NotImplementedError

    cpdef find_host(self, DTYPE_FLOAT_t xpos_old, DTYPE_FLOAT_t ypos_old,
            DTYPE_FLOAT_t xpos_new, DTYPE_FLOAT_t ypos_new, DTYPE_INT_t guess):
        raise NotImplementedError

    cpdef find_host_using_global_search(self, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos):
        raise NotImplementedError

    cpdef find_host_using_local_search(self, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t first_guess):
        raise NotImplementedError

    cpdef get_boundary_intersection(self, DTYPE_FLOAT_t xpos_old,
            DTYPE_FLOAT_t ypos_old, DTYPE_FLOAT_t xpos_new,
            DTYPE_FLOAT_t ypos_new, DTYPE_INT_t last_host):
        raise NotImplementedError

    def set_local_coordinates_wrapper(self, ParticleSmartPtr particle):
        return self.set_local_coordinates(particle.get_ptr())

    cdef set_local_coordinates(self, Particle *particle):
        raise NotImplementedError

    def set_vertical_grid_vars_wrapper(self, DTYPE_FLOAT_t time,
                                       ParticleSmartPtr particle):
        return self.set_vertical_grid_vars(time, particle.get_ptr())

    cdef set_vertical_grid_vars(self, DTYPE_FLOAT_t time, Particle *particle):
        raise NotImplementedError

    def get_zmin_wrapper(self, DTYPE_FLOAT_t time, ParticleSmartPtr particle):
        return self.get_zmin(time, particle.get_ptr())

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle):
        raise NotImplementedError

    def get_zmax_wrapper(self, DTYPE_FLOAT_t time, ParticleSmartPtr particle):
        return self.get_zmax(time, particle.get_ptr())

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle):
        raise NotImplementedError

    cdef get_velocity(self, DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t vel[3]):
        raise NotImplementedError

    cdef get_horizontal_velocity(self, DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t vel[2]):
        raise NotImplementedError
    
    cdef get_vertical_velocity(self, DTYPE_FLOAT_t time, Particle *particle):
        raise NotImplementedError

    def get_horizontal_eddy_viscosity_wrapper(self, DTYPE_FLOAT_t time,
                                              ParticleSmartPtr particle):
        return self.get_horizontal_eddy_viscosity(time, particle.get_ptr())

    cdef get_horizontal_eddy_viscosity(self, DTYPE_FLOAT_t time,
            Particle *particle):
        raise NotImplementedError

    cdef get_horizontal_eddy_viscosity_derivative(self, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t Ah_prime[2]):
        raise NotImplementedError

    def get_vertical_eddy_diffusivity_wrapper(self, DTYPE_FLOAT_t time,
                                              ParticleSmartPtr particle):
        return self.get_vertical_eddy_diffusivity(time, particle.get_ptr())

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            Particle *particle) except FLOAT_ERR:
        raise NotImplementedError

    def get_vertical_eddy_diffusivity_derivative_wrapper(self, DTYPE_FLOAT_t time,
                                                         ParticleSmartPtr particle):
        return self.get_vertical_eddy_diffusivity_derivative(time, particle.get_ptr())

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self, 
            DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        raise NotImplementedError
    
    cpdef DTYPE_INT_t is_wet(self, DTYPE_FLOAT_t time, DTYPE_INT_t host) except INT_ERR:
        raise NotImplementedError
