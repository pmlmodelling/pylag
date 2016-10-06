import numpy as np

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from data_reader cimport DataReader

cimport interpolation as interp

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

    # Sea surface elevation
    cdef DTYPE_FLOAT_t _zeta_last
    cdef DTYPE_FLOAT_t _zeta_next

    # Z level depths
    cdef DTYPE_FLOAT_t[:] _zlev_last, _zlev_next

    # Eddy diffusivity on depth levels
    cdef DTYPE_FLOAT_t[:] _kh_last, _kh_next   

    # Time
    cdef DTYPE_FLOAT_t _time_last
    cdef DTYPE_FLOAT_t _time_next

    def __init__(self, config, mediator):
        self.config = config
        self.mediator = mediator
        
        self._n_zlay = self.mediator.get_dimension_variable('z')
        self._n_zlev = self._n_zlay + 1

        # Initialise as empty arrays - these are set to meaningful values
        # through calls to _read_time_dependent_vars.
        self._zlev_last = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._zlev_next = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._kh_last = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._kh_next = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        

    cdef _read_time_dependent_vars(self):
        """ Update time variables and memory views for GOTM data fields
        
        For each GOTM time-dependent variable needed by PyLag two references
        are stored. These correspond to the last and next time points at which
        GOTM data was saved. Together these bound PyLag's current time point.
        
        All communications go via the mediator in order to guarentee support for
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
        self._zeta_last = self.mediator.get_time_dependent_variable_at_last_time_index('zeta', (1), DTYPE_FLOAT)[0]
        self._zeta_next = self.mediator.get_time_dependent_variable_at_next_time_index('zeta', (1), DTYPE_FLOAT)[0]

        # Update memory views for interface depths at the last time point
        h = self.mediator.get_time_dependent_variable_at_last_time_index('h', (self._n_zlay), DTYPE_FLOAT)
        z = self.mediator.get_time_dependent_variable_at_last_time_index('z', (self._n_zlay), DTYPE_FLOAT)
        for i in xrange(self._n_zlay):
            self._zlev_last[i] = z[i] - h[i]/2.0
        self._zlev_last[-1] = z[i] + h[i]/2.0

        # Update memory views for interface depths at the next time point
        h = self.mediator.get_time_dependent_variable_at_next_time_index('h', (self._n_zlay), DTYPE_FLOAT)
        z = self.mediator.get_time_dependent_variable_at_next_time_index('z', (self._n_zlay), DTYPE_FLOAT)
        for i in xrange(self._n_zlay):
            self._zlev_next[i] = z[i] - h[i]/2.0
        self._zlev_next[-1] = z[-1] + h[-1]/2.0
        
        # Update memory views for kh. As GOTM drops the value of the diffusivity
        # at the last interface this must be set separately.
        kh_last = self.mediator.get_time_dependent_variable_at_last_time_index('nuh', (self._n_zlay), DTYPE_FLOAT)
        kh_next = self.mediator.get_time_dependent_variable_at_next_time_index('nuh', (self._n_zlay), DTYPE_FLOAT)
        for i in xrange(self._n_zlay):
            self._kh_last[i+1] = kh_last[i]
            self._kh_next[i+1] = kh_next[i]

        # At the missing interface set the diffusivity to zero
        self._kh_last[0] = 0.0
        self._kh_next[0] = 0.0
        
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
        is the case, arrays containing time-dependent variable data are updated.
        
        Parameters:
        -----------
        time : float
            The current time.
        """
        time_fraction = interp.get_linear_fraction(time, self._time_last, self._time_next)
        if time_fraction < 0.0 or time_fraction >= 1.0:
            self.mediator.update_reading_frames(time)
            self._read_time_dependent_vars()

    cpdef DTYPE_INT_t find_host(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos,
        DTYPE_INT_t guess):
        return 0

    cpdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos):
        """ Returns zmin

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
        zmin : float
            z min.
        """
        cdef DTYPE_FLOAT_t time_fraction
        
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        return interp.linear_interp(time_fraction, self._zlev_last[0], self._zlev_next[0])

    cpdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos):
        """ Returns zmax

        Parameters:
        -----------
        time : float
            Time.

        xpos : float
            x-position.

        ypos : float
            y-position
        
        Returns:
        --------
        zmax : float
            z max.
        """
        cdef DTYPE_FLOAT_t time_fraction
        
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)
        
        return interp.linear_interp(time_fraction, self._zlev_last[-1], self._zlev_next[-1])

    cpdef get_bathymetry(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, 
            DTYPE_INT_t host):
        """ Returns the column depth
        
        Parameters:
        -----------
        xpos : float
            x-position at which to interpolate (unused).
        
        ypos : float
            y-position at which to interpolate (unused).
            
        host : int
            Host horizontal element (unused).

        Return:
        -------
        h : float
            Bathymetry.
        """
        return -self._zlev_last[0]

    cpdef get_sea_sur_elev(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t host):
        """ Returns the sea surface elevation through linear interpolation.
        
        Interpolation proceeds through linear interpolation in time.
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate (unused).
        
        ypos : float
            y-position at which to interpolate (unused).
            
        host : int
            Host horizontal element (unused).

        Return:
        -------
        zeta : float
            Sea surface elevation.
        """
        cdef DTYPE_FLOAT_t time_fraction
        
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)
        
        return interp.linear_interp(time_fraction, self._zeta_last, self._zeta_next)
    
    cpdef get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host):
        """ Returns the vertical eddy diffusivity through linear interpolation.
        
        The vertical eddy diffusivity is defined at layer interfaces.
        Interpolation is performed first in time, then in z.
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate (unused).
        
        ypos : float
            y-position at which to interpolate (unused).

        zpos : float
            z-position at which to interpolate.

        host : int
            Host horizontal element (unused).
        
        Returns:
        --------
        kh : float
            The vertical eddy diffusivity.        
        
        """
        # Variables used when determining indices for the z levels that
        # bound the particle's position
        cdef DTYPE_INT_t k_lower_level, k_upper_level
        cdef DTYPE_FLOAT_t z_lower_level, z_upper_level        
        cdef bint particle_found
        
        # Time and z fractions for interpolation in time and z
        cdef DTYPE_FLOAT_t time_fraction, z_fraction
        
        # Interpolated diffusivities on lower and upper bounding z levels
        cdef DTYPE_FLOAT_t kh_lower_level
        cdef DTYPE_FLOAT_t kh_upper_level
        
        # Get time fraction for interpolation in time
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)
        
        # Determine upper and lower bounding z levels
        particle_found = False
        for i in xrange(self._n_zlay):
            k_lower_level = i
            k_upper_level = i+1
            z_lower_level = interp.linear_interp(time_fraction, self._zlev_last[k_lower_level], self._zlev_next[k_lower_level])
            z_upper_level = interp.linear_interp(time_fraction, self._zlev_last[k_upper_level], self._zlev_next[k_upper_level])
            
            if zpos <= z_upper_level and zpos >= z_lower_level:
                particle_found = True
                break
        
        if particle_found is False:
            raise ValueError("Particle zpos (={} not found!".format(zpos))

        # Interpolate kh in time
        kh_lower_level = interp.linear_interp(time_fraction, self._kh_last[k_lower_level], self._kh_next[k_lower_level])
        kh_upper_level = interp.linear_interp(time_fraction, self._kh_last[k_upper_level], self._kh_next[k_upper_level])
        
        # Interpolate kh in z
        z_fraction = interp.get_linear_fraction_safe(zpos, z_lower_level, z_upper_level)
        return interp.linear_interp(z_fraction, kh_lower_level, kh_upper_level)

    cpdef get_vertical_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t time,
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos,
            DTYPE_INT_t host):
        """ Returns the gradient in the vertical eddy diffusivity.
        
        Return a numerical approximation of the gradient in the vertical eddy 
        diffusivity at (t,x,y,z).
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        xpos : float
            x-position at which to interpolate (unused).
        
        ypos : float
            y-position at which to interpolate (unused).

        zpos : float
            z-position at which to interpolate.

        host : int
            Host horizontal element (unused).
        
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
        
        # Use a point arbitrarily close to zpos (in sigma coordinates) for the 
        # gradient calculation
        zpos_increment = 1.0e-3
        
        # Use the negative of zpos_increment at the top of the water column
        if ((zpos + zpos_increment) > 0.0):
            zpos_increment = -zpos_increment
            
        # A point close to zpos
        zpos_incremented = zpos + zpos_increment

        # Compute the gradient
        k1 = self.get_vertical_eddy_diffusivity(time, xpos, ypos, zpos, host)
        k2 = self.get_vertical_eddy_diffusivity(time, xpos, ypos, zpos_incremented, host)
        k_prime = (k2 - k1) / zpos_increment

        return k_prime
