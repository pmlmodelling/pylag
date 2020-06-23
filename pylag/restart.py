"""
Restart module that provides the following functionality:

1) Writing of model restart files

The reading of restart files is implemented in `particle_initialisation.py`.
"""

import os
import logging
import numpy as np
from netCDF4 import Dataset, date2num

from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag import variable_library


class RestartFileCreator(object):
    """ Restart file creator

    Objects of type RestartFileCreator manage the creation of
    model restart files.

    Parameters:
    -----------
    config : SafeConfigParser
        Configuration obect.
    """

    def __init__(self, config):
        # Configuration object
        self._config = config

        # The directory in which restart files will be created
        self._restart_dir = self._config.get("RESTART", "restart_dir")

        # Read in the coordinate system
        coordinate_system = self._config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system").strip().lower()
        if coordinate_system in ["cartesian", "spherical"]:
            self.coordinate_system = coordinate_system
        else:
            raise ValueError("Unsupported model coordinate system `{}'".format(coordinate_system))

        if not os.path.isdir(self._restart_dir):
            logger = logging.getLogger(__name__)

            logger.info('Creating restart file directory {}.'.format(self._restart_dir))
            os.mkdir(self._restart_dir)

    def create(self, filename_stem, n_particles, datetime, particle_data):
        """ Create a restart file

        Restart files will be created in the directory `restart_dir`. The following
        file naming convention is used:

        <restart_dir>/<filename_stem>_YYYYMMDD-HHMMSS.nc

        The time stamp is obtained from the supplied datetime parameter.

        All the particle data that is required to restart the model should
        be provided in the dictionary `particle_data`. This allows for a
        completely generic structure.

        Parameters:
        -----------
        filename_str : str
            The file name stem, as explaing above (e.g. `set_1`).

        n_particles : int
            The total number of particles.

        datetime : Datetime
            Datetime object corresponding to the current date and time.

        particle_data : dict
            Dictionary containing particle data.
        """
        logger = logging.getLogger(__name__)

        # Compute the time stamp
        time_stamp = datetime.strftime('%Y%m%d-%H%M%S')

        # Construct the file name from the filename stem and time stamp
        file_name = '{}/{}_{}.nc'.format(self._restart_dir, filename_stem, time_stamp)

        # Open the restart file for writing
        try:
            logger.info('Creating restart file {} at time {}.'.format(file_name, time_stamp))
            nc_file = Dataset(file_name, mode='w', format='NETCDF4')
        except:
            logger.error('Failed to create restart file: {}.'.format(file_name))
            raise

        # Variable names
        variable_names = particle_data.keys()

        # Create coordinate variables etc.
        self._create_file_structure(nc_file, n_particles, variable_names)

        # Save the current time
        nc_file.variables['time'][0] = date2num(datetime, units=nc_file.variables['time'].units)

        # Save variable data
        for var_name in variable_names:
            if var_name in ['x1', 'x2', 'x3']:
                var_name_key = variable_library.get_coordinate_variable_name(self.coordinate_system, var_name)
            else:
                var_name_key = var_name
            nc_file.variables[var_name_key][0, :] = particle_data[var_name]

        # Close the file
        nc_file.close()

        return

    def _create_file_structure(self, nc_file, n_particles, variable_names):
        """ Create NetCDF dimension and particle data variables

        Parameters:
        -----------
        nc_file : NetCDF4.Dataset
            The restart dataset

        n_particles : int
            The number of particles

        variable_names : list[str]
            List of variables names to be saved

        """

        # Compression options for the netCDF variables.
        ncopts = {'zlib': True, 'complevel': 7}

        # Global attributes
        nc_file.title = 'PyLag restart file'

        # Create coordinate dimensions
        nc_file.createDimension('particles', n_particles)
        nc_file.createDimension('time', 1)

        # Add time variable
        time = nc_file.createVariable('time', DTYPE_INT, ('time',))
        time.units = 'seconds since 1960-01-01 00:00:00'
        time.calendar = 'standard'
        time.long_name = 'Time'

        # Add particle variables
        for var_name in variable_names:
            if var_name in ['x1', 'x2', 'x3']:
                var_name = variable_library.get_coordinate_variable_name(self.coordinate_system, var_name)
            data_type = variable_library.get_data_type(var_name)
            units = variable_library.get_units(var_name)
            long_name = variable_library.get_long_name(var_name)

            var = nc_file.createVariable(var_name, data_type, ('time', 'particles',), **ncopts)
            var.units = units
            var.long_name = long_name

        return


