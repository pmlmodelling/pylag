"""
Abstract base class for PyLag DataReaders.
"""

include "constants.pxi"

import numpy as np

from pylag.particle_cpp_wrapper cimport ParticleSmartPtr

cdef class DataReader:
    """ Abstract base class for PyLag data readers

    PyLag DataReaders provide a common interface through which it is possible
    to access input data that is used to drive the particle tracking model. The
    desired input data is returned in a common format, irrespective of its
    origin. By subclassing PyLag DataReader, it becomes possible
    to extend the model to accept input data in many different formats.
    Examples of input data include fluid velocity vectors and eddy
    diffusivities. These may come from an analytical model, or be defined
    on a discrete numerical grid, as used by a particular hydrodynamic of
    ocean general circulation model; all of these can be supported by
    subclassing DataReader and implementing code that extracts and returns
    the required information.

    For efficiency reasons, DataReader has been implemented in Cython and
    only part of its API is exposed in Python. In order to make it possible
    to use DataReader objects in Python, a set of Python wrappers have been
    added to the DataReader base class. These are documented here.
    """
    cpdef setup_data_access(self, start_datetime, end_datetime):
        """ Setup access to time dependant variables

        Setup access to input data using the supplied datatime arguments.

        Parameters
        ----------
        start_datetime : datetime.datetime
            The simulation state time/date

        end_datetime : datetime.datetime
            The simulation end time/date
        """
        raise NotImplementedError

    cpdef read_data(self, DTYPE_FLOAT_t time):
        """ Read in time dependent variable data from file

        `time` is used to test if new data should be read in from file.

        Parameters
        ----------
        time : float
            The current time.
        """
        raise NotImplementedError

    cpdef get_grid_names(self):
        """ Return a list of grid names on which data are defined

        Returns
        -------
         : list
             A list of grid names
        """
        return []

    def find_host_wrapper(self, ParticleSmartPtr particle_old,
                          ParticleSmartPtr particle_new):
        """ Python wrapper for finding and setting the particle's host element

        Parameters
        ----------
        particle_old : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its last known position.

        particle_new : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its new position.

        Returns
        -------
         : int
             Flag signifying whether the new host element was found successfully.
        """
        return self.find_host(particle_old.get_ptr(), particle_new.get_ptr())

    cdef DTYPE_INT_t find_host(self, Particle *particle_old,
                               Particle *particle_new) except INT_ERR:
        raise NotImplementedError

    def find_host_using_global_search_wrapper(self,
                                              ParticleSmartPtr particle):
        """ Python wrapper for finding and setting the host element using a global search

        Parameters
        ----------
        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : int
             Flag signifying whether the host element was found successfully.
        """
        return self.find_host_using_global_search(particle.get_ptr())

    cdef DTYPE_INT_t find_host_using_global_search(self,
                                                   Particle *particle) except INT_ERR:
        raise NotImplementedError

    def find_host_using_local_search_wrapper(self,
                                             ParticleSmartPtr particle):
        """ Python wrapper for finding and setting the host element using a local search

        Parameters
        ----------
        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : int
             Flag signifying whether the host element was found successfully.
        """
        return self.find_host_using_local_search(particle.get_ptr())

    cdef DTYPE_INT_t find_host_using_local_search(self,
                                                  Particle *particle) except INT_ERR:
        raise NotImplementedError

    def get_boundary_intersection_wrapper(self, ParticleSmartPtr particle_old,
                                  ParticleSmartPtr particle_new):
        """ Python wrapper for finding the point no the boundary the particle intersected

        This can be used when imposing horizontal boundary conditions.

        Parameters
        ----------
        particle_old : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its last known position.

        particle_new : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its new position.

        Returns
        -------
        start_point : NumPy array
            Start coordinates of the side the particle crossed.

        end_point : NumPy array
            End coordinates of the side the particle crossed.

        intersection : NumPy array
            Coordinates of the intersection point.

        """
        cdef DTYPE_FLOAT_t start_point_c[2]
        cdef DTYPE_FLOAT_t end_point_c[2]
        cdef DTYPE_FLOAT_t intersection_c[2]

        self.get_boundary_intersection(particle_old.get_ptr(), particle_new.get_ptr(), start_point_c, end_point_c, intersection_c)

        start_point, end_point, intersection = [], [], []
        for i in range(2):
            start_point.append(start_point_c[i])
            end_point.append(end_point_c[i])
            intersection.append(intersection_c[i])

        return np.array(start_point), np.array(end_point), np.array(intersection)

    cdef get_boundary_intersection(self,
                                   Particle *particle_old,
                                   Particle *particle_new,
                                   DTYPE_FLOAT_t start_point[2],
                                   DTYPE_FLOAT_t end_point[2],
                                   DTYPE_FLOAT_t intersection[2]):
        raise NotImplementedError

    def set_default_location_wrapper(self, ParticleSmartPtr particle):
        """ Python wrapper for setting the default location of a particle

        Can be used to set a particle's position to a default location
        within an element (for example, should it not be possible to
        strictly apply the horizontal boundary condition).

        Parameters
        ----------
        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : None
        """
        return self.set_default_location(particle.get_ptr())

    cdef set_default_location(self, Particle *particle):
        raise NotImplementedError

    def set_local_coordinates_wrapper(self, ParticleSmartPtr particle):
        """ Python wrapper for setting particle local coordinates

        Used to set particle local horizontal coordinates (e.g. within its host
        element).

        Parameters
        ----------
        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : None
        """
        return self.set_local_coordinates(particle.get_ptr())

    cdef set_local_coordinates(self, Particle *particle):
        raise NotImplementedError

    def set_vertical_grid_vars_wrapper(self, DTYPE_FLOAT_t time,
                                       ParticleSmartPtr particle):
        """ Python wrapper for setting particle vertical grid variables

        Vertical grid variables include local vertical coordinates and
        indices identifying the current vertical level within which the
        particle resides.

        Parameters
        ----------
        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : int
             Flag signifying whether or not the operation was a success.
        """
        return self.set_vertical_grid_vars(time, particle.get_ptr())

    cdef DTYPE_INT_t set_vertical_grid_vars(self, DTYPE_FLOAT_t time,
                                            Particle *particle) except INT_ERR:
        raise NotImplementedError

    cpdef DTYPE_FLOAT_t get_xmin(self) except FLOAT_ERR:
        """ Python wrapper for getting the minimum `x` value across the grid

        Primarily introduced to increase the number of significant figures used
        when computing changes in particle positions on small domains.

        Returns
        -------
         : float
             The minimum `x` value
        """
        raise NotImplementedError

    cpdef DTYPE_FLOAT_t get_ymin(self) except FLOAT_ERR:
        """ Python wrapper for getting the minimum `y` value across the grid

        Primarily introduced to increase the number of significant figures used
        when computing changes in particle positions on small domains.

        Returns
        -------
         : float
             The minimum `y` value
        """
        raise NotImplementedError

    def get_zmin_wrapper(self, DTYPE_FLOAT_t time, ParticleSmartPtr particle):
        """ Python wrapper for getting the minimum `z` value at the particle's position

        Typically, the minimum z-value will correspond to the bathymetry at the given time
        and the particle's current position.

        Parameters
        ----------
        time : float
            The current time.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : float
             The minimum `z` value
        """
        return self.get_zmin(time, particle.get_ptr())

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        raise NotImplementedError

    def get_zmax_wrapper(self, DTYPE_FLOAT_t time, ParticleSmartPtr particle):
        """ Python wrapper for getting the maximum `z` value at the particle's position

        Typically, the maximum z-value will correspond to the free surface height at the given time
        and the particle's current position.

        Parameters
        ----------
        time : float
            The current time.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : float
             The maximum `z` value
        """
        return self.get_zmax(time, particle.get_ptr())

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        raise NotImplementedError

    def get_velocity_wrapper(self, DTYPE_FLOAT_t time, ParticleSmartPtr particle,
                             vel):
        """ Python wrapper for getting the velocity at the particle's position

        This function will return a three component vector giving the fluid velocity
        at the given time and the particle's current location. The velocity is passed
        back through the supplied argument `vel`.

        Parameters
        ----------
        time : float
            The current time.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        vel : 1D NumPy array of length three.
            The velocity vector.

        Returns
        -------
         : None
        """
        cdef DTYPE_FLOAT_t vel_c[3]

        if len(vel.shape) != 1 or vel.shape[0] != 3:
            raise ValueError('Invalid vel array')

        vel_c[:] = vel[:]

        self.get_velocity(time, particle.get_ptr(), vel_c)

        for i in xrange(3):
            vel[i] = vel_c[i]

        return

    cdef void get_velocity(self, DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t vel[3]) except +:
        raise NotImplementedError

    def get_horizontal_velocity_wrapper(self, DTYPE_FLOAT_t time, ParticleSmartPtr particle,
                                        vel):
        """ Python wrapper for getting the horizontal velocity at the particle's position

        This function will return a two component vector giving the fluid velocity
        at the given time and the particle's current location. The velocity is passed
        back through the supplied argument `vel`.

        Parameters
        ----------
        time : float
            The current time.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        vel : 1D NumPy array of length two.
            The velocity vector.

        Returns
        -------
         : None
        """
        cdef DTYPE_FLOAT_t vel_c[2]

        if len(vel.shape) != 1 or vel.shape[0] != 2:
            raise ValueError('Invalid vel array')

        vel_c[:] = vel[:]

        self.get_velocity(time, particle.get_ptr(), vel_c)

        for i in xrange(2):
            vel[i] = vel_c[i]

        return

    cdef void get_horizontal_velocity(self, DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t vel[2]) except +:
        raise NotImplementedError
    
    cdef DTYPE_FLOAT_t get_vertical_velocity(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        raise NotImplementedError

    def get_horizontal_eddy_viscosity_wrapper(self, DTYPE_FLOAT_t time,
                                              ParticleSmartPtr particle):
        """ Python wrapper for getting the horizontal eddy viscosity at the particle's position

        Parameters
        ----------
        time : float
            The current time.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        vel : 1D NumPy array of length two.
            The velocity vector.

        Returns
        -------
         : float
            The horizontal eddy viscosity
        """
        return self.get_horizontal_eddy_viscosity(time, particle.get_ptr())

    cdef DTYPE_FLOAT_t get_horizontal_eddy_viscosity(self, DTYPE_FLOAT_t time,
            Particle *particle) except FLOAT_ERR:
        raise NotImplementedError

    def get_horizontal_eddy_viscosity_derivative_wrapper(self, DTYPE_FLOAT_t time,
                                                          ParticleSmartPtr particle,
                                                          Ah_prime):
        """ Python wrapper for getting the horizontal eddy viscosity at the particle's position

        Parameters
        ----------
        time : float
            The current time.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Ah_prime : 1D NumPy array of length two.
            The derivative of the horizontal eddy viscosity

        Returns
        -------
         : None
        """
        cdef DTYPE_FLOAT_t Ah_prime_c[2]

        if len(Ah_prime.shape) != 1 or Ah_prime.shape[0] != 2:
            raise ValueError('Invalid Ah_prime array')

        Ah_prime_c[:] = Ah_prime[:]

        self.get_horizontal_eddy_viscosity_derivative(time, particle.get_ptr(), Ah_prime_c)

        for i in xrange(2):
            Ah_prime[i] = Ah_prime_c[i]

        return

    cdef void get_horizontal_eddy_viscosity_derivative(self, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t Ah_prime[2]) except +:
        raise NotImplementedError

    def get_vertical_eddy_diffusivity_wrapper(self, DTYPE_FLOAT_t time,
                                              ParticleSmartPtr particle):
        """ Python wrapper for getting the vertical eddy diffusivity at the particle's position

        Parameters
        ----------
        time : float
            The current time.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : float
            The vertical eddy diffusivity
        """
        return self.get_vertical_eddy_diffusivity(time, particle.get_ptr())

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            Particle *particle) except FLOAT_ERR:
        raise NotImplementedError

    def get_vertical_eddy_diffusivity_derivative_wrapper(self, DTYPE_FLOAT_t time,
                                                         ParticleSmartPtr particle):
        """ Python wrapper for getting the vertical eddy diffusivity derivative at the particle's position

        Parameters
        ----------
        time : float
            The current time.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : float
            The vertical eddy diffusivity
        """
        return self.get_vertical_eddy_diffusivity_derivative(time, particle.get_ptr())

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self, 
            DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        raise NotImplementedError
    
    def is_wet_wrapper(self, DTYPE_FLOAT_t time, ParticleSmartPtr particle):
        """ Python wrapper for getting the wet/dry status of the particle

        Parameters
        ----------
        time : float
            The current time.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : int
            Flag identifying whether the current cell is wet or dry
        """
        return self.is_wet(time, particle.get_ptr())

    cdef DTYPE_INT_t is_wet(self, DTYPE_FLOAT_t time, Particle *particle) except INT_ERR:
        raise NotImplementedError

    def get_environmental_variable_wrapper(self, str var_name, DTYPE_FLOAT_t time,
                                    ParticleSmartPtr particle):
        """ Python wrapper for getting environmental variables

        Parameters
        ----------
        var_name : str
            The variable name

        time : float
            The time

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            The particle at its current position.

        Returns
        -------
         : float
            The value of the environmental variable at the particle's current position
        """
        return self.get_environmental_variable(var_name, time, particle.get_ptr())

    cdef DTYPE_FLOAT_t get_environmental_variable(self, var_name, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        raise NotImplementedError

