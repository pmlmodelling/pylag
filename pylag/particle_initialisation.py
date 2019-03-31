""" Particle initialiser

This module provides the following functionality:

1) Reading of particle initialisation files
2) Reading of model restart files
3) Writing of model restart files
"""

import numpy as np
import logging
from netCDF4 import Dataset, num2date
import datetime

from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT


class InitialParticleStateReader(object):
    """ Initial particle state reader

    Abstract base class for initial particle state readers. Such
    objects are are used to read or calculate initial particle
    state data such as initial positions or in different contexts.
    These are then returned to the caller in the form of lists. The
    caller must manage the actual setting of particle properties for
    each Particle object it manages - this is not done here.
    """
    def get_particle_data(self):
        raise NotImplementedError


class ASCIIInitialParticleStateReader(InitialParticleStateReader):
    """ ASCII initial particle state reader

    ASCIIInitialParticleStateReaders read in particle state data 
    from an ascii file and return it to the caller.

    TODO - At the moment, such objects only read in particle position info.
           It may be desirable to have them read other types of data in the
           future.
    """
    def __init__(self, config):
        self._config = config

        self.file_name = config.get('SIMULATION', 'initial_positions_file')

    def get_particle_data(self):
        """ Get particle data

        Particle data is read in from an ASCII file.
        """
        # Input file containing particle initial positions
        with open(self.file_name, 'r') as f:
            lines = f.readlines()

        data = []
        for line in lines:
            data.append(line.rstrip('\r\n').split(' '))

            # The first entry is the number of particles
        n_particles = self._get_entry(data.pop(0)[0], DTYPE_INT)

        # Create seed particle set
        group_ids = []
        x_positions = []
        y_positions = []
        z_positions = []
        for row in data:
            group_ids.append(self._get_entry(row[0], DTYPE_INT))
            x_positions.append(self._get_entry(row[1], DTYPE_FLOAT))
            y_positions.append(self._get_entry(row[2], DTYPE_FLOAT))
            z_positions.append(self._get_entry(row[3], DTYPE_FLOAT))

        group_ids = np.array(group_ids)
        x_positions = np.array(x_positions)
        y_positions = np.array(y_positions)
        z_positions = np.array(z_positions)

        return n_particles, group_ids, x_positions, y_positions, z_positions

    def _get_entry(self, value, type):
        return type(value)


class RestartInitialParticleStateReader(InitialParticleStateReader):
    """ Restart initial particle state reader

    RestartInitialParticleStateReaders read in particle state data from a
    restart file in NetCDF format.
    """
    def __init__(self, config):
        self._config = config

        self._restart_file_name = self._config.get('RESTART', 'restart_file_name')

    def get_particle_data(self):
        """ Get particle data

        Particle data is read in from a NetCDF file that has been created
        using an object of type RestartFileCreator.

        Parameters:
        -----------
        file_name : str
            The name of the NetCDF file containing particle state data.
        """
        logger = logging.getLogger(__name__)
        logger.info('Using restart file {}'.format(self._restart_file_name))

        # Open the file for reading
        try:
            restart = Dataset(self._restart_file_name, 'r')
            logger.info('Opened data file {} for reading.'.format(self._restart_file_name))
        except Exception:
            logger.error('Failed to open restart file {}.'.format(self._restart_file_name))
            raise

        # Check time
        datetime_start_str = self._config.get("SIMULATION", "start_datetime")
        datetime_start = datetime.datetime.strptime(datetime_start_str, "%Y-%m-%d %H:%M:%S")
        datetime_restart = num2date(restart.variables['time'][0],
                                    units=restart.variables['time'].units,
                                    calendar=restart.variables['time'].calendar)

        if datetime_start != datetime_restart:
            datetime_restart_str = datetime_restart.strftime("%Y-%m-%d %H:%M:%S")
            logger.error("The specified start time `{}' and restart time `{}' do not match".format(datetime_start_str, datetime_restart_str))
            raise ValueError("When restarting the model, the specified start time should match that given in the restart file.")

        # Extract particle data
        n_particles = restart.dimensions['particles'].size
        group_ids = restart.variables['group_id'][0,:]
        x_positions = restart.variables['xpos'][0,:]
        y_positions = restart.variables['ypos'][0,:]
        z_positions = restart.variables['zpos'][0,:]

        restart.close()

        return n_particles, group_ids, x_positions, y_positions, z_positions


def get_initial_particle_state_reader(config):
    if config.get("SIMULATION", "initialisation_method") == "init_file":
        return ASCIIInitialParticleStateReader(config)
    elif config.get("SIMULATION", "initialisation_method") == "restart_file":
        return RestartInitialParticleStateReader(config)
    elif config.get("SIMULATION", "initialisation_method") == "rectangular_grid":
        raise NotImplementedError
    else:
        raise ValueError('Unrecognised initialisation method {}'.format(initialisation_method))