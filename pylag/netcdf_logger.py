"""
Module containing classes that facilitate the writing of data to file
"""

import logging
import numpy as np
from netCDF4 import Dataset
from cftime import num2pydate
from cftime import date2num
import pathlib

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_python import INT_INVALID, FLOAT_INVALID

from pylag import variable_library
from pylag.exceptions import PyLagValueError


class NetCDFLogger(object):
    """ NetCDF data logger.

    Objects of type NetCDFLogger can be used to write particle data to file.

    Parameters
    ----------
    config : ConfigParser
        Configuration object

    file_name : str
        Name of the `*.nc` output file to be generated. If the `.nc` extension
        is not present it is automatically added.

    start_datetime : str
        String giving the simulation start date and time.
    
    n_particles : int
        The number of particles.

    grid_names : list[str]
        List of grids on which input data is defined.
    """
    def __init__(self, config, file_name, start_datetime, n_particles,
                 grid_names):
        
        logger = logging.getLogger(__name__)

        self.config = config

        if file_name[-3:] == '.nc':
            self.file_name = file_name
        else:
            self.file_name = ''.join([file_name, '.nc'])

        pathlib.Path(self.file_name).unlink(missing_ok=True)

        logger.info(f'Creating output file: {self.file_name}.')
        self._ncfile = Dataset(self.file_name, mode='w', format='NETCDF4')

        # Compression options for the netCDF variables.
        self._ncopts = {'zlib': True, 'complevel': 7}

        # Time units
        self._simulation_time_units = f'seconds since {start_datetime}'

        # Read in the coordinate system
        coordinate_system = self.config.get("SIMULATION",
                                            "coordinate_system").strip().lower()
        if coordinate_system in ["cartesian", "geographic"]:
            self.coordinate_system = coordinate_system
        else:
            raise PyLagValueError(f"Unsupported model coordinate "
                                  f"system `{coordinate_system}'")

        # Are we saving biological variables
        try:
            self.with_biology = self.config.getboolean("BIO_MODEL",
                                                       "use_bio_model")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self.with_biology = False

        # Set precision for diagnostic variables
        try:
            precision = self.config.get("OUTPUT", "precision")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            logger.info(f'The precision of outputs has not been specified. '
                        f'Defaulting to single precision.')
            precision = "single"

        if precision == "single":
            self._precision = 's'
        elif precision == "double":
            self._precision = 'd'
        else:
            raise PyLagValueError(f'Unknown precision parameter `{precision}')

        # Local environmental variables
        self._env_vars = {}

        # Add environmental variables, as requested
        try:
            var_names = self.config.get("OUTPUT",
                    "environmental_variables").strip().split(',')
            self.environmental_variables = \
                    [var_name.strip() for var_name in var_names]
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self.environmental_variables = []

        # Save a list of extra grid variables to be returned as diagnostics
        try:
            var_names = self.config.get("OUTPUT",
                    "extra_grid_variables").strip().split(',')
            self.extra_grid_variables = \
                    [var_name.strip() for var_name in var_names]
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self.extra_grid_variables = []

        # Grid names
        self.grid_names = grid_names

        # Host horizontal elements
        self._host_vars = {}

        # Create coordinate variables etc.
        self._create_file_structure(n_particles)
        
    def _create_file_structure(self, n_particles):

        self._ncfile.title = 'PyLag -- Plymouth Marine Laboratory'

        # Create coordinate dimensions
        self._ncfile.createDimension('particles', n_particles)
        self._ncfile.createDimension('time', None)
 
        # Add time variable
        self._time = self._ncfile.createVariable('time', np.int64, ('time',))
        self._time.units = 'seconds since 1960-01-01 00:00:00'
        self._time.calendar = 'standard'
        self._time.long_name = 'Time'

        # Add particle group ids
        self._group_id = self._ncfile.createVariable('group_id',
                variable_library.get_integer_type(self._precision),
                ('particles',))
        self._group_id.long_name = 'Particle group ID'

        # x1
        x1_var_name = variable_library.get_coordinate_variable_name(
                self.coordinate_system, 'x1')
        self._x1 = self._ncfile.createVariable(x1_var_name,
                variable_library.get_data_type(x1_var_name, self._precision),
                ('time', 'particles',), **self._ncopts)
        self._x1.units = variable_library.get_units(x1_var_name)
        self._x1.long_name = variable_library.get_long_name(x1_var_name)

        # x2
        x2_var_name = variable_library.get_coordinate_variable_name(
                self.coordinate_system, 'x2')
        self._x2 = self._ncfile.createVariable(x2_var_name,
                variable_library.get_data_type(x2_var_name, self._precision),
                ('time', 'particles',), **self._ncopts)
        self._x2.units = variable_library.get_units(x2_var_name)
        self._x2.long_name = variable_library.get_long_name(x2_var_name)

        # x3
        x3_var_name = variable_library.get_coordinate_variable_name(
                self.coordinate_system, 'x3')
        self._x3 = self._ncfile.createVariable(x3_var_name,
                variable_library.get_data_type(x3_var_name, self._precision),
                ('time', 'particles',), **self._ncopts)
        self._x3.units = variable_library.get_units(x3_var_name)
        self._x3.long_name = variable_library.get_long_name(x3_var_name)

        # Add hosts
        for grid_name in self.grid_names:
            self._host_vars[grid_name] = self._ncfile.createVariable(
                    f'host_{grid_name}',
                    variable_library.get_integer_type(self._precision),
                    ('time', 'particles',), **self._ncopts)
            self._host_vars[grid_name].units = 'None'
            self._host_vars[grid_name].long_name = (f"Host horizontal element "
                                                    f"on grid {grid_name}")
            self._host_vars[grid_name].invalid = f"{INT_INVALID}"
        
        # Add status variables
        data_type = variable_library.get_data_type('error_status',
                                                   self._precision)
        self._status = self._ncfile.createVariable('error_status', data_type,
                ('time', 'particles',), **self._ncopts)
        self._status.units = variable_library.get_units('error_status')
        self._status.long_name = variable_library.get_long_name('error_status')

        data_type = variable_library.get_data_type('in_domain', self._precision)
        self._in_domain = self._ncfile.createVariable('in_domain', data_type,
                ('time', 'particles',), **self._ncopts)
        self._in_domain.units = variable_library.get_units('in_domain')
        self._in_domain.long_name = variable_library.get_long_name('in_domain')

        data_type = variable_library.get_data_type('is_beached',
                                                   self._precision)
        self._is_beached = self._ncfile.createVariable('is_beached', data_type,
                ('time', 'particles',), **self._ncopts)
        self._is_beached.units = variable_library.get_units('is_beached')
        self._is_beached.long_name = \
                variable_library.get_long_name('is_beached')

        # Extra grid variables
        if 'h' in self.extra_grid_variables:
            data_type = variable_library.get_data_type('h', self._precision)
            self._h = self._ncfile.createVariable('h', data_type,
                    ('time', 'particles',), **self._ncopts)
            self._h.units = variable_library.get_units('h')
            self._h.long_name = variable_library.get_long_name('h')
            self._h.invalid = f'{variable_library.get_invalid_value(data_type)}'

        if 'zeta' in self.extra_grid_variables:
            data_type = variable_library.get_data_type('zeta', self._precision)
            self._zeta = self._ncfile.createVariable('zeta', data_type,
                    ('time', 'particles',), **self._ncopts)
            self._zeta.units = variable_library.get_units('zeta')
            self._zeta.long_name = variable_library.get_long_name('zeta')
            self._zeta.invalid = \
                    f'{variable_library.get_invalid_value(data_type)}'

        # Add number of land boundary encounters
        data_type = variable_library.get_data_type('land_boundary_encounters',
                                                   self._precision)
        self._land_boundary_encounters = self._ncfile.createVariable(
                'land_boundary_encounters', data_type,
                ('time', 'particles',), **self._ncopts)
        self._land_boundary_encounters.units = \
            variable_library.get_units('land_boundary_encounters')
        self._land_boundary_encounters.long_name = \
            variable_library.get_long_name('land_boundary_encounters')

        # Add environmental variables
        for var_name in self.environmental_variables:
            data_type = variable_library.get_data_type(var_name,
                                                       self._precision)
            units = variable_library.get_units(var_name)
            long_name = variable_library.get_long_name(var_name)
            invalid = variable_library.get_invalid_value(data_type)

            self._env_vars[var_name] = self._ncfile.createVariable(var_name,
                    data_type, ('time', 'particles',), **self._ncopts)
            self._env_vars[var_name].units = units
            self._env_vars[var_name].long_name = long_name
            self._env_vars[var_name].invalid = f'{invalid}'

        # Bio model variables
        if self.with_biology:
            data_type = variable_library.get_data_type('age', self._precision)
            self._age = self._ncfile.createVariable('age',
                    data_type, ('time', 'particles',), **self._ncopts)
            self._age.units = variable_library.get_units('age')
            self._age.long_name = variable_library.get_long_name('age')
            self._age.invalid = \
                    f"{variable_library.get_invalid_value(data_type)}"

            data_type = variable_library.get_data_type('is_alive',
                                                       self._precision)
            self._is_alive = self._ncfile.createVariable('is_alive',
                    data_type, ('time', 'particles',), **self._ncopts)
            self._is_alive.units = variable_library.get_units('is_alive')
            self._is_alive.long_name = \
                    variable_library.get_long_name('is_alive')

    def write_group_ids(self, group_ids):
        """ Write particle group IDs to file

        Parameters
        ----------
        group_ids : array_like
            Particle group IDs.

        Returns
        -------
        None
        """
        self._group_id[:] = group_ids

    def write(self, time, particle_data):
        """ Write particle data to file

        Parameters
        ----------
        time : float
            The current time

        particle_data : dict
            Dictionary containing particle data with format
            ['attribute': [...]].

        Returns
        -------
        None
        """
        # Next time index
        tidx = self._time.shape[0]
        
        # Rebase time units and save
        dt = num2pydate(time, units=self._simulation_time_units)
        self._time[tidx] = date2num(dt, units=self._time.units)
        
        self._x1[tidx, :] = particle_data['x1']
        self._x2[tidx, :] = particle_data['x2']
        self._x3[tidx, :] = particle_data['x3']
        self._is_beached[tidx, :] = particle_data['is_beached']
        self._in_domain[tidx, :] = particle_data['in_domain']
        self._status[tidx, :] = particle_data['error_status']
        self._land_boundary_encounters[tidx, :] = \
            particle_data['land_boundary_encounters']

        # Add host horizontal elements
        for grid_name in self.grid_names:
            self._host_vars[grid_name][tidx, :] = \
                    particle_data[f'host_{grid_name}']

        # Add extra grid variables
        if 'h' in self.extra_grid_variables:
            self._h[tidx, :] = particle_data['h']
        if 'zeta' in self.extra_grid_variables:
            self._zeta[tidx, :] = particle_data['zeta']

        # Add environmental variables
        for var_name in self.environmental_variables:
            self._env_vars[var_name][tidx, :] = particle_data[var_name]

        # Add bio model variables
        if self.with_biology:
            self._age[tidx, :] = particle_data['age']
            self._is_alive[tidx, :] = particle_data['is_alive']

    def sync(self):
        """ Sync data to disk

        Returns
        -------
        None
        """
        # Sync data to disk
        self._ncfile.sync()
        
    def close(self):
        """ Close the logger

        Returns
        -------
        None
        """
        logger = logging.getLogger(__name__)
        logger.info('Closing data logger.')
        self._ncfile.close()


__all__ = ['NetCDFLogger']
