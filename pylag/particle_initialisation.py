""" Particle initialiser

This module provides the following functionality:

1) Reading of particle initialisation files
2) Reading of model restart files
3) Writing of model restart files
"""

import os
import logging
import numpy as np
from netCDF4 import Dataset, date2num

from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag import variable_library


def read_particle_initial_positions(file_name):
    """
    Read in particle initial positions from file and return to caller.
    
    Parameters
    ----------
    file_name: string
        Name of the file.
        
    Returns
    -------
    group_id: list, int
        List of particle group ids
    xpos: list, float
        List of x position
    ypos: list, float
        List of y positions
    zpos: list, float
        List of z positions (either cartesian or sigma coordinates)
    """
    
    # Input file containing particle initial positions
    with open(file_name, 'r') as f:
        lines=f.readlines()

    data=[]
    for line in lines:
        data.append(line.rstrip('\r\n').split(' '))    
        
    # The first entry is the number of particles
    n_particles = _get_entry(data.pop(0)[0], DTYPE_INT)

    # Create seed particle set
    group_ids = []
    x_positions = []
    y_positions = []
    z_positions = []
    for row in data:
        group_ids.append(_get_entry(row[0], DTYPE_INT))
        x_positions.append(_get_entry(row[1], DTYPE_FLOAT))
        y_positions.append(_get_entry(row[2], DTYPE_FLOAT))
        z_positions.append(_get_entry(row[3], DTYPE_FLOAT))
    
    group_ids = np.array(group_ids)
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    z_positions = np.array(z_positions)
    
    return n_particles, group_ids, x_positions, y_positions, z_positions


def _get_entry(value, type):
    return type(value)


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
            nc_file.variables[var_name][0, :] = particle_data[var_name]

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
            data_type = variable_library.get_data_type(var_name)
            units = variable_library.get_units(var_name)
            long_name = variable_library.get_long_name(var_name)

            var = nc_file.createVariable(var_name, data_type, ('time', 'particles',), **ncopts)
            var.units = units
            var.long_name = long_name

        return

