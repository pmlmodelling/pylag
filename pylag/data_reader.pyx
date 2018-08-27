include "constants.pxi"

from pylag.particle cimport ParticleSmartPtr

cdef class DataReader:
    cpdef setup_data_access(self, start_datetime, end_datetime):
        raise NotImplementedError

    cpdef read_data(self, DTYPE_FLOAT_t time):
        raise NotImplementedError

    def find_host_wrapper(self, ParticleSmartPtr particle_old,
                          ParticleSmartPtr particle_new):
        return self.find_host(particle_old.get_ptr(), particle_new.get_ptr())

    cdef DTYPE_INT_t find_host(self, Particle *particle_old,
                               Particle *particle_new) except INT_ERR:
        raise NotImplementedError

    def find_host_using_global_search_wrapper(self,
                                              ParticleSmartPtr particle):
        return self.find_host_using_global_search(particle.get_ptr())

    cdef DTYPE_INT_t find_host_using_global_search(self,
                                                   Particle *particle) except INT_ERR:
        raise NotImplementedError

    def find_host_using_local_search_wrapper(self,
                                             ParticleSmartPtr particle,
                                             DTYPE_INT_t first_guess):
        return self.find_host_using_local_search(particle.get_ptr(),
                                                 first_guess)

    cdef DTYPE_INT_t find_host_using_local_search(self,
                                                  Particle *particle,
                                                  DTYPE_INT_t first_guess) except INT_ERR:
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

    def get_velocity_wrapper(self, DTYPE_FLOAT_t time, ParticleSmartPtr particle,
                             vel):
        cdef DTYPE_FLOAT_t vel_c[3]

        if len(vel.shape) != 1 or vel.shape[0] != 3:
            raise ValueError('Invalid vel array')

        vel_c[:] = vel[:]

        self.get_velocity(time, particle.get_ptr(), vel_c)

        for i in xrange(3):
            vel[i] = vel_c[i]

        return

    cdef get_velocity(self, DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t vel[3]):
        raise NotImplementedError

    def get_horizontal_velocity_wrapper(self, DTYPE_FLOAT_t time, ParticleSmartPtr particle,
                                        vel):
        cdef DTYPE_FLOAT_t vel_c[2]

        if len(vel.shape) != 1 or vel.shape[0] != 2:
            raise ValueError('Invalid vel array')

        vel_c[:] = vel[:]

        self.get_velocity(time, particle.get_ptr(), vel_c)

        for i in xrange(2):
            vel[i] = vel_c[i]

        return

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

    def get_horizontal_eddy_viscosity_derivative_wrapper(self, DTYPE_FLOAT_t time,
                                                          ParticleSmartPtr particle,
                                                          Ah_prime):
        cdef DTYPE_FLOAT_t Ah_prime_c[2]

        if len(Ah_prime.shape) != 1 or Ah_prime.shape[0] != 2:
            raise ValueError('Invalid Ah_prime array')

        Ah_prime_c[:] = Ah_prime[:]

        self.get_horizontal_eddy_viscosity_derivative(time, particle.get_ptr(), Ah_prime_c)

        for i in xrange(2):
            Ah_prime[i] = Ah_prime_c[i]

        return

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
