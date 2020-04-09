include "constants.pxi"

import numpy as np

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from pylag.particle cimport Particle

from pylag.data_reader cimport DataReader

from pylag.math cimport sigma_to_cartesian_coords

cimport pylag.interpolation as interp
import pylag.interpolation as interp

from pylag import variable_library
from pylag.numerics import get_time_direction

cdef class GOTMDataReader(DataReader):
    """ DataReader for GOTM.
    
    Objects of type GOTMDataReader are intended to manage all access to GOTM 
    data objects, including data describing the model grid as well as model
    output variables.
    
    GOTMDataReader employs linear interpolation in time. For interpolation
    in space, GOTM uses a dedicated Interpolator. The choice of interpolator
    is made at run time. See PyLag's documentation for a list of supported
    interpolation methods.
    
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

    # List of environmental variables to read and save
    cdef object env_var_names

    # Number of z levels and layers
    cdef DTYPE_INT_t _n_zlay, _n_zlev

    # Time direction
    cdef DTYPE_INT_t _time_direction

    # Time
    cdef DTYPE_FLOAT_t _time_last, _time_next, _time_fraction

    # Water depth (from the mean sea surface height to the sea floor)
    cdef DTYPE_FLOAT_t _H

    # Sea surface elevation
    cdef DTYPE_FLOAT_t _zeta_last, _zeta_next, _zeta

    # Z layer depths
    cdef DTYPE_FLOAT_t[:] _zlay_last, _zlay_next, _zlay

    # Z level depths
    cdef DTYPE_FLOAT_t[:] _zlev_last, _zlev_next, _zlev

    # Eddy diffusivity at layer interfaces
    cdef DTYPE_FLOAT_t[:] _kh_last, _kh_next, _kh
    
    # Eddy diffusivity derivative at layer interfaces
    cdef DTYPE_FLOAT_t[:] _kh_prime

    # Temperature
    cdef DTYPE_FLOAT_t[:] _thetao_last, _thetao_next, _thetao

    # Salinity
    cdef DTYPE_FLOAT_t[:] _so_last, _so_next, _so

    # Short wave downwelling irradiance
    cdef DTYPE_FLOAT_t[:] _rsdo_last, _rsdo_next, _rsdo

    # Interpolator
    cdef interp.Interpolator _kh_interpolator
    cdef interp.Interpolator _thetao_interpolator
    cdef interp.Interpolator _so_interpolator
    cdef interp.Interpolator _rsdo_interpolator

    def __init__(self, config, mediator):
        self.config = config
        self.mediator = mediator
        
        self._time_direction = <int>get_time_direction(config)

        self._n_zlay = self.mediator.get_dimension_variable('z')
        self._n_zlev = self.mediator.get_dimension_variable('zi')

        # Initialise as empty arrays - these are set to meaningful values
        # through calls to _read_time_dependent_vars.
        self._zlay_last = np.empty((self._n_zlay), dtype=DTYPE_FLOAT)
        self._zlay_next = np.empty((self._n_zlay), dtype=DTYPE_FLOAT)
        self._zlay = np.empty((self._n_zlay), dtype=DTYPE_FLOAT)
        self._zlev_last = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._zlev_next = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._zlev = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._kh_last = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._kh_next = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._kh = np.empty((self._n_zlev), dtype=DTYPE_FLOAT)
        self._thetao_last = np.empty((self._n_zlay), dtype=DTYPE_FLOAT)
        self._thetao_next = np.empty((self._n_zlay), dtype=DTYPE_FLOAT)
        self._thetao = np.empty((self._n_zlay), dtype=DTYPE_FLOAT)
        self._so_last = np.empty((self._n_zlay), dtype=DTYPE_FLOAT)
        self._so_next = np.empty((self._n_zlay), dtype=DTYPE_FLOAT)
        self._so = np.empty((self._n_zlay), dtype=DTYPE_FLOAT)
        self._rsdo_last = np.empty((self._n_zlay), dtype=DTYPE_FLOAT)
        self._rsdo_next = np.empty((self._n_zlay), dtype=DTYPE_FLOAT)
        self._rsdo = np.empty((self._n_zlay), dtype=DTYPE_FLOAT)

        # Interpolator
        self._kh_interpolator = interp.get_interpolator(self.config, self._n_zlev)
        self._thetao_interpolator = interp.get_interpolator(self.config, self._n_zlay)
        self._so_interpolator = interp.get_interpolator(self.config, self._n_zlay)
        self._rsdo_interpolator = interp.get_interpolator(self.config, self._n_zlay)

        # Check to see if any environmental variables are being saved.
        try:
            env_var_names = self.config.get("OUTPUT", "environmental_variables").strip().split(',')
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            env_var_names = []

        self.env_var_names = []
        for env_var_name in env_var_names:
            env_var_name = env_var_name.strip()
            if env_var_name is not None:
                if env_var_name in variable_library.gotm_variable_names.keys():
                    self.env_var_names.append(env_var_name)
                else:
                    raise ValueError('Received unsupported variable {}'.format(env_var_name))

        self._read_time_dependent_vars()

    cdef _read_time_dependent_vars(self):
        """ Update time variables and memory views for GOTM data fields
        
        For each GOTM time-dependent variable needed by PyLag two references
        are stored. These correspond to the last and next time points at which
        GOTM data was saved. Together these bound PyLag's current time point.
        
        All communications go via the mediator in order to guarantee support for
        both serial and parallel simulations.
        
        Parameters:
        -----------
        N/A
        
        Returns:
        --------
        N/A
        """
        # Update time references
        self._time_last = self.mediator.get_time_at_last_time_index()
        self._time_next = self.mediator.get_time_at_next_time_index()

        # Update zeta
        self._zeta_last = self.mediator.get_time_dependent_variable_at_last_time_index('zeta', (1,1), DTYPE_FLOAT)[0,0]
        self._zeta_next = self.mediator.get_time_dependent_variable_at_next_time_index('zeta', (1,1), DTYPE_FLOAT)[0,0]

        # Update z layers
        self._zlay_last = self.mediator.get_time_dependent_variable_at_last_time_index('z', (self._n_zlay,1,1), DTYPE_FLOAT)[:,0,0]
        self._zlay_next = self.mediator.get_time_dependent_variable_at_next_time_index('z', (self._n_zlay,1,1), DTYPE_FLOAT)[:,0,0]

        # Update z levels
        self._zlev_last = self.mediator.get_time_dependent_variable_at_last_time_index('zi', (self._n_zlev,1,1), DTYPE_FLOAT)[:,0,0]
        self._zlev_next = self.mediator.get_time_dependent_variable_at_next_time_index('zi', (self._n_zlev,1,1), DTYPE_FLOAT)[:,0,0]
        
        # Update memory views for kh
        self._kh_last = self.mediator.get_time_dependent_variable_at_last_time_index('nuh', (self._n_zlev,1,1), DTYPE_FLOAT)[:,0,0]
        self._kh_next = self.mediator.get_time_dependent_variable_at_next_time_index('nuh', (self._n_zlev,1,1), DTYPE_FLOAT)[:,0,0]

        # Read in data as requested
        if 'thetao' in self.env_var_names:
            gotm_var_name = variable_library.gotm_variable_names['thetao']
            self._thetao_last = self.mediator.get_time_dependent_variable_at_last_time_index(gotm_var_name, (self._n_zlay, 1, 1), DTYPE_FLOAT)[:, 0, 0]
            self._thetao_next = self.mediator.get_time_dependent_variable_at_next_time_index(gotm_var_name, (self._n_zlay, 1, 1), DTYPE_FLOAT)[:, 0, 0]

        if 'so' in self.env_var_names:
            gotm_var_name = variable_library.gotm_variable_names['so']
            self._so_last = self.mediator.get_time_dependent_variable_at_last_time_index(gotm_var_name, (self._n_zlay, 1, 1), DTYPE_FLOAT)[:, 0, 0]
            self._so_next = self.mediator.get_time_dependent_variable_at_next_time_index(gotm_var_name, (self._n_zlay, 1, 1), DTYPE_FLOAT)[:, 0, 0]

        if 'rsdo' in self.env_var_names:
            gotm_var_name = variable_library.gotm_variable_names['rsdo']
            self._rsdo_last = self.mediator.get_time_dependent_variable_at_last_time_index(gotm_var_name, (self._n_zlay, 1, 1), DTYPE_FLOAT)[:, 0, 0]
            self._rsdo_next = self.mediator.get_time_dependent_variable_at_next_time_index(gotm_var_name, (self._n_zlay, 1, 1), DTYPE_FLOAT)[:, 0, 0]

        return

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

        self._kh_interpolator.set_points(self._zlev, self._kh)

        for i in xrange(self._n_zlay):
            self._zlay[i] = interp.linear_interp(self._time_fraction, self._zlay_last[i], self._zlay_next[i])

        # Read in data as requested
        if 'thetao' in self.env_var_names:
            for i in xrange(self._n_zlay):
                self._thetao[i] = interp.linear_interp(self._time_fraction, self._thetao_last[i], self._thetao_next[i])

            self._thetao_interpolator.set_points(self._zlay, self._thetao)

        if 'so' in self.env_var_names:
            for i in xrange(self._n_zlay):
                self._so[i] = interp.linear_interp(self._time_fraction, self._so_last[i], self._so_next[i])

            self._so_interpolator.set_points(self._zlay, self._so)

        if 'rsdo' in self.env_var_names:
            for i in xrange(self._n_zlay):
                self._rsdo[i] = interp.linear_interp(self._time_fraction, self._rsdo_last[i], self._rsdo_next[i])

            self._rsdo_interpolator.set_points(self._zlay, self._rsdo)

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
        if self._time_direction == 1:
            if time_fraction < 0.0 or time_fraction >= 1.0:
                self.mediator.update_reading_frames(time)
                self._read_time_dependent_vars()
        else:
            if time_fraction <= 0.0 or time_fraction > 1.0:
                self.mediator.update_reading_frames(time)
                self._read_time_dependent_vars()
            
        self._interpolate_in_time(time)

    cdef DTYPE_INT_t find_host(self, Particle *particle_old,
                               Particle *particle_new) except INT_ERR:
        return IN_DOMAIN

    cdef DTYPE_INT_t find_host_using_global_search(self,
                                                   Particle *particle) except INT_ERR:
        return IN_DOMAIN

    cdef DTYPE_INT_t find_host_using_local_search(self,
                                                  Particle *particle_old) except INT_ERR:
        return IN_DOMAIN

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

    cdef DTYPE_INT_t set_vertical_grid_vars(self, DTYPE_FLOAT_t time,
                                            Particle *particle) except INT_ERR:
        """ Set variables describing the particle's position in z
        
        Find the host vertical layer. Begin with a local search using the old
        z layer as a starting point. If this fails search the full vertical grid.
        """
        cdef DTYPE_INT_t k
        cdef DTYPE_FLOAT_t z

        # Start with a local search
        for k in xrange(particle.get_k_layer()-2, particle.get_k_layer()+2, 1):
            if k < 0 or k >= self._n_zlay:
                continue

            if particle.get_x3() <= self._zlev[k+1] and particle.get_x3() >= self._zlev[k]:
                particle.set_k_layer(k)
                particle.set_omega_interfaces(interp.get_linear_fraction_safe(particle.get_x3(), self._zlev[k], self._zlev[k+1]))
                return IN_DOMAIN

        # Search the full vertical grid
        for k in xrange(self._n_zlay): 
            if particle.get_x3() <= self._zlev[k+1] and particle.get_x3() >= self._zlev[k]:
                particle.set_k_layer(k)
                particle.set_omega_interfaces(interp.get_linear_fraction_safe(particle.get_x3(), self._zlev[k], self._zlev[k+1]))
                return IN_DOMAIN
    
        # Return error flag if particle not found
        return BDY_ERROR

    cpdef DTYPE_FLOAT_t get_xmin(self) except FLOAT_ERR:
        return 0.0

    cpdef DTYPE_FLOAT_t get_ymin(self) except FLOAT_ERR:
        return 0.0

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        """ Returns the column depth

        Parameters:
        -----------
        time : float
            Time.

        x1 : float
            x-position (unused).

        x2 : float
            y-position (unused).
        
        Returns:
        --------
        H : float
            The column depth.
        """
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)

        return interp.linear_interp(time_fraction, self._zlev_last[0], self._zlev_next[0])

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
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

        x1 : float
            x-position.

        x2 : float
            y-position
        
        Return:
        -------
        zeta : float
            Sea surface elevation.
        """
        time_fraction = interp.get_linear_fraction_safe(time, self._time_last, self._time_next)
        
        return interp.linear_interp(self._time_fraction, self._zeta_last, self._zeta_next)

    cdef get_velocity(self, DTYPE_FLOAT_t time, Particle* particle,
            DTYPE_FLOAT_t vel[3]):
        """ Returns the velocity u(t,x,y,z)
        
        For now, simply return a zeroed array.

        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.

        Return:
        -------
        vel : C array, float
            u/v/w velocity components stored in a C array.           
        """
        # Compute u/v velocities and save
        vel[0] = 0.0
        vel[1] = 0.0
        vel[2] = 0.0
        return
    
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
        cdef DTYPE_FLOAT_t value

        value = self._kh_interpolator.get_value(particle)
        if value < 0.0:
            return 0.0
        return value

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
        return self._kh_interpolator.get_first_derivative(particle)

    cdef DTYPE_FLOAT_t get_environmental_variable(self, var_name,
            DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        """ Returns the value of the given environmental variable

        Support for extracting the following GOTM environmental variables has been implemented:

        thetao - Sea water potential temperature

        so - Sea water salinty

        rsdo - Short wave downwelling irradiance

        Parameters:
        -----------
        var_name : str
            The name of the variable. See above for a list of supported options.

        time : float
            Time at which to interpolate.

        particle: *Particle
            Pointer to a Particle object.

        Returns:
        --------
        var : float
            The interpolated value of the variable at the specified point in time and space.
        """
        cdef DTYPE_FLOAT_t value # Environmental variable at (t, x3)

        if var_name in self.env_var_names:
            if var_name == 'thetao':
                value = self._thetao_interpolator.get_value(particle)
            elif var_name == 'so':
                value = self._so_interpolator.get_value(particle)
            elif var_name == 'rsdo':
                value = self._rsdo_interpolator.get_value(particle)
            return value
        else:
            raise ValueError("Received unsupported environmental variable `{}'".format(var_name))

    cdef DTYPE_INT_t is_wet(self, DTYPE_FLOAT_t time, Particle *particle) except INT_ERR:
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

