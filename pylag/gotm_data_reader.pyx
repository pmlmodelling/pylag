include "constants.pxi"

import numpy as np

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from particle cimport Particle

from pylag.data_reader cimport DataReader

from pylag.math cimport sigma_to_cartesian_coords

cimport pylag.interpolation as interp

cdef class GOTMDataReader(DataReader):
    """ DataReader for GOTM.
    
    Objects of type GOTMDataReader are intended to manage all access to GOTM 
    data objects, including data describing the model grid as well as model
    output variables.
    
    Parameters:
    -----------
    config : SafeConfigParser
        Configuration object.
    
    mediator : Mediator
        Mediator object for managing access to data read from file.
    """
    
    # Configurtion object
    cdef object config

    # Mediator for accessing GOTM model data read in from file
    cdef object mediator

    # Number of z levels and layers
    cdef DTYPE_INT_t _n_zlay, _n_zlev

    # Time
    cdef DTYPE_FLOAT_t _time_last, _time_next, _time_fraction

    # Water depth (from the mean sea surface height to the sea floor)
    cdef DTYPE_FLOAT_t _H

    # Sea surface elevation
    cdef DTYPE_FLOAT_t _zeta_last, _zeta_next, _zeta

    # Z level depths
    cdef DTYPE_FLOAT_t[:] _zlev_last, _zlev_next, _zlev

    # Eddy diffusivity on depth levels
    cdef DTYPE_FLOAT_t[:] _kh_last, _kh_next, _kh

    def __init__(self, config, mediator):
        self.config = config
        self.mediator = mediator
        
        self._n_zlay = self.mediator.get_dimension_variable('z')
        self._n_zlev = self._n_zlay + 1

        # Initialise as empty arrays - these are set to meaningful values
        # through calls to _read_time_dependent_vars.
        self._zlev_last = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._zlev_next = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._zlev = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._kh_last = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._kh_next = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._kh = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        
    cdef _read_time_dependent_vars(self):
        """ Update time variables and memory views for GOTM data fields
        
        For each GOTM time-dependent variable needed by PyLag two references
        are stored. These correspond to the last and next time points at which
        GOTM data was saved. Together these bound PyLag's current time point.
        
        All communications go via the mediator in order to guarantee support for
        both serial and parallel simulations.
        
        Note that GOTM discards data for the bottom interface when saving data 
        to file. To account for this the depth of z levels is set using the 
        depth of layer centres and layer widths while kh is set equal to zero at
        the missing interface.
        
        Parameters:
        -----------
        N/A
        
        Returns:
        --------
        N/A
        """
        cdef DTYPE_FLOAT_t[:] h, z
        cdef DTYPE_FLOAT_t[:] kh_last, kh_next
        cdef DTYPE_INT_t i
        
        # Update time references
        self._time_last = self.mediator.get_time_at_last_time_index()
        self._time_next = self.mediator.get_time_at_next_time_index()

        # Update zeta
        self._zeta_last = self.mediator.get_time_dependent_variable_at_last_time_index('zeta', (1,1), DTYPE_FLOAT)[0,0]
        self._zeta_next = self.mediator.get_time_dependent_variable_at_next_time_index('zeta', (1,1), DTYPE_FLOAT)[0,0]

        # Update memory views for interface depths at the last time point
        h = self.mediator.get_time_dependent_variable_at_last_time_index('h', (self._n_zlay,1,1), DTYPE_FLOAT)[:,0,0]
        z = self.mediator.get_time_dependent_variable_at_last_time_index('z', (self._n_zlay,1,1), DTYPE_FLOAT)[:,0,0]
        for i in xrange(self._n_zlay):
            self._zlev_last[i] = z[i] - h[i]/2.0
        self._zlev_last[-1] = z[i] + h[i]/2.0

        # Update memory views for interface depths at the next time point
        h = self.mediator.get_time_dependent_variable_at_next_time_index('h', (self._n_zlay,1,1), DTYPE_FLOAT)[:,0,0]
        z = self.mediator.get_time_dependent_variable_at_next_time_index('z', (self._n_zlay,1,1), DTYPE_FLOAT)[:,0,0]
        for i in xrange(self._n_zlay):
            self._zlev_next[i] = z[i] - h[i]/2.0
        self._zlev_next[-1] = z[-1] + h[-1]/2.0
        
        # Update memory views for kh. As GOTM drops the value of the diffusivity
        # at the last interface this must be set separately.
        kh_last = self.mediator.get_time_dependent_variable_at_last_time_index('nuh', (self._n_zlay,1,1), DTYPE_FLOAT)[:,0,0]
        kh_next = self.mediator.get_time_dependent_variable_at_next_time_index('nuh', (self._n_zlay,1,1), DTYPE_FLOAT)[:,0,0]
        for i in xrange(self._n_zlay):
            self._kh_last[i+1] = kh_last[i]
            self._kh_next[i+1] = kh_next[i]

        # At the missing interface set the diffusivity to zero
        self._kh_last[0] = 0.0
        self._kh_next[0] = 0.0

    cdef _interpolate_in_time(self, time):
        """ Linearly interpolate in time all time dependent variables
        
        Interpolate all gridded variables in time and store for later use. This
        saves having to compute the same quantities at the same instance in time
        for each particle. This results in much reduced run times for the case
        in which n_particles >> n_levels.
        """
        cdef DTYPE_INT_t i
        
        self._time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        self._H = -interp.linear_interp(self._time_fraction, self._zlev_last[0], self._zlev_next[0])
        
        self._zeta = interp.linear_interp(self._time_fraction, self._zeta_last, self._zeta_next)

        for i in xrange(self._n_zlev): 
            self._zlev[i] = interp.linear_interp(self._time_fraction, self._zlev_last[i], self._zlev_next[i])

            self._kh[i] = interp.linear_interp(self._time_fraction, self._kh_last[i], self._kh_next[i])
        
    cpdef setup_data_access(self, start_datetime, end_datetime):
        """ Set up access to time-dependent variables.
        
        Parameters:
        -----------
        start_datetime : Datetime
            Datetime object corresponding to the simulation start time.
        
        end_datetime : Datetime
            Datetime object corresponding to the simulation end time.
        """
        self.mediator.setup_data_access(start_datetime, end_datetime)

        self._read_time_dependent_vars()

    cpdef read_data(self, DTYPE_FLOAT_t time):
        """ Read in time dependent variable data from file?
        
        `time' is used to test if new data should be read in from file. If this
        is the case arrays containing time-dependent variable data are updated.
        Following this, all time interpolated gridded fields are computed.

        Parameters:
        -----------
        time : float
            The current time.
        """
        time_fraction = interp.get_linear_fraction(time, self._time_last, self._time_next)
        if time_fraction < 0.0 or time_fraction >= 1.0:
            self.mediator.update_reading_frames(time)
            self._read_time_dependent_vars()
        
        self._interpolate_in_time(time)

    cpdef find_host(self, DTYPE_FLOAT_t xpos_old, DTYPE_FLOAT_t ypos_old,
            DTYPE_FLOAT_t xpos_new, DTYPE_FLOAT_t ypos_new, DTYPE_INT_t guess):
        return IN_DOMAIN, DEFAULT_HOST

    cpdef find_host_using_global_search(self, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos):
        return DEFAULT_HOST

    cpdef find_host_using_local_search(self, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t first_guess):
        return IN_DOMAIN, DEFAULT_HOST

    cdef set_local_coordinates(self, Particle *particle):
        """ Set local coordinates
        
        Each particle has associated with it a set of global coordinates
        and a set of local coordinates. Here, the global coordinates and the 
        host horizontal element are used to set the local coordinates.
        
        Parameters:
        -----------
        particle: *Particle
            Pointer to a Particle struct
        """
        pass

    cdef set_vertical_grid_vars(self, DTYPE_FLOAT_t time, Particle* particle):
        """ Set variables describing the particle's position in z
        
        Find the host vertical layer. Begin with a local search using the old
        z layer as a starting point. If this fails search the full vertical grid.
        """
        cdef DTYPE_INT_t k
        cdef DTYPE_FLOAT_t z

        # Start with a local search
        for k in xrange(particle.k_layer-2, particle.k_layer+2, 1):
            if k < 0 or k >= self._n_zlay:
                continue

            if particle.zpos <= self._zlev[k+1] and particle.zpos >= self._zlev[k]:
                particle.k_layer = k
                particle.omega_interfaces = interp.get_linear_fraction_safe(particle.zpos, self._zlev[k], self._zlev[k+1])
                return

        # Search the full vertical grid
        for k in xrange(self._n_zlay): 
            if particle.zpos <= self._zlev[k+1] and particle.zpos >= self._zlev[k]:
                particle.k_layer = k
                particle.omega_interfaces = interp.get_linear_fraction_safe(particle.zpos, self._zlev[k], self._zlev[k+1])
                return
    
        # Search failed
        raise ValueError("Particle z position (={}) not found!".format(particle.zpos))

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle):
        """ Returns the column depth

        Parameters:
        -----------
        time : float
            Time.

        xpos : float
            x-position (unused).

        ypos : float
            y-position (unused).
        
        Returns:
        --------
        H : float
            The column depth.
        """
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        return interp.linear_interp(time_fraction, self._zlev_last[0], self._zlev_next[0])

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle):
        """ Returns zmax in cartesian coordinates

        Returns the stored sea surface elevation that was set from the last
        call to `read_data'. All function arguments are ignored meaning
        interpolation is time is not performed! Please call `read_data' before
        calling this function in order to update GotmDataReader's internal
        state.

        Parameters:
        -----------
        time : float (unused)
            Time.

        xpos : float
            x-position.

        ypos : float
            y-position
        
        Return:
        -------
        zeta : float
            Sea surface elevation.
        """
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)
        
        return interp.linear_interp(self._time_fraction, self._zeta_last, self._zeta_next)
    
    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time,
            Particle *particle) except FLOAT_ERR:
        """ Returns the vertical eddy diffusivity through linear interpolation.
        
        The vertical eddy diffusivity is defined at layer interfaces.
        Interpolation is performed first in time, then in z.
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.
        
        Returns:
        --------
        kh : float
            The vertical eddy diffusivity.        
        
        """
        return interp.linear_interp(particle.omega_interfaces, self._kh[particle.k_layer], self._kh[particle.k_layer+1])

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self,
            DTYPE_FLOAT_t time, Particle* particle) except FLOAT_ERR:
        """ Returns the gradient in the vertical eddy diffusivity.
        
        Return a numerical approximation of the gradient in the vertical eddy 
        diffusivity at (t,x,y,z).
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        particle : *Particle
            Pointer to a Particle object.
        
        Returns:
        --------
        k_prime : float
            Gradient in the vertical eddy diffusivity field.
        """
        # Diffusivities
        cdef DTYPE_FLOAT_t kh1, kh2
        
        # Diffusivity gradient
        cdef DTYPE_FLOAT_t k_prime
        
        # Z coordinate vars for the gradient calculation
        cdef DTYPE_FLOAT_t zpos_increment, zpos_incremented

        # Z layer for incremented position
        cdef DTYPE_INT_t zlayer_incremented

        # Temporary particle object
        cdef Particle _particle

        # Create a copy of the particle
        _particle = particle[0]

        # Compute the diffusivity at the current particle's location
        kh1 = self.get_vertical_eddy_diffusivity(time, &_particle)

        # Find a point near to zpos
        zpos_increment = 1.0e-3

        # ... use the negative of zpos_increment at the top of the water column
        if ((_particle.zpos + zpos_increment) > self._zeta):
            zpos_increment = -zpos_increment

        _particle.zpos = _particle.zpos + zpos_increment
        self.set_vertical_grid_vars(time, &_particle)

        kh2 = self.get_vertical_eddy_diffusivity(time, &_particle)
        k_prime = (kh2 - kh1) / zpos_increment

        return k_prime

    cpdef DTYPE_INT_t is_wet(self, DTYPE_FLOAT_t time, DTYPE_INT_t host) except INT_ERR:
        """ Return an integer indicating whether `host' is wet or dry
        
        The function returns 1 if `host' is wet at time `time' and 
        0 if `host' is dry.
        
        For GOTM, we assume there is always water present.
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.

        host : int
            Integer that identifies the host element in question
        """
        return 1