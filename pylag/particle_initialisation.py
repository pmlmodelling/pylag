"""
Particle state initialisation

This module provides the following functionality:

1) Reading of particle initialisation files
2) Reading of model restart files
"""

import numpy as np
import logging
from netCDF4 import Dataset
from cftime import num2pydate
import datetime

from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT

from pylag.exceptions import PyLagValueError
from pylag import variable_library


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

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration object

    TODO
    ----
    * At the moment, such objects only read in particle position info.
    It may be desirable to have them read other types of data in the future.
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
        x1_positions = []
        x2_positions = []
        x3_positions = []
        for row in data:
            group_ids.append(self._get_entry(row[0], DTYPE_INT))
            x1_positions.append(self._get_entry(row[1], DTYPE_FLOAT))
            x2_positions.append(self._get_entry(row[2], DTYPE_FLOAT))
            x3_positions.append(self._get_entry(row[3], DTYPE_FLOAT))

        group_ids = np.array(group_ids)
        x1_positions = np.array(x1_positions)
        x2_positions = np.array(x2_positions)
        x3_positions = np.array(x3_positions)

        return n_particles, group_ids, x1_positions, x2_positions, x3_positions

    def _get_entry(self, value, type):
        return type(value)


class RestartInitialParticleStateReader(InitialParticleStateReader):
    """ Restart initial particle state reader

    RestartInitialParticleStateReaders read in particle state data from a
    restart file in NetCDF format.

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration object
    """
    def __init__(self, config):
        self._config = config

        self._restart_file_name = self._config.get('RESTART',
                                                   'restart_file_name')

        # Read in the coordinate system
        coordinate_system = self.config.get("SIMULATION",
                                            "coordinate_system").strip().lower()
        if coordinate_system in ["cartesian", "geographic"]:
            self.coordinate_system = coordinate_system
        else:
            raise PyLagValueError(f"Unsupported model coordinate "
                                  f"system `{coordinate_system}`")

    def get_particle_data(self):
        """ Get particle data

        Particle data is read in from a NetCDF file that has been created
        using an object of type RestartFileCreator.
        """
        logger = logging.getLogger(__name__)
        logger.info(f'Using restart file {self._restart_file_name}')

        # Open the file for reading
        try:
            restart = Dataset(self._restart_file_name, 'r')
            logger.info(f'Opened data file {self._restart_file_name} '
                        f'for reading.')
        except Exception:
            logger.error(f'Failed to open restart file '
                         f'{self._restart_file_name}.')
            raise

        # Check time
        datetime_start_str = self._config.get("SIMULATION", "start_datetime")
        datetime_start = datetime.datetime.strptime(datetime_start_str,
                                                    "%Y-%m-%d %H:%M:%S")
        datetime_restart = num2pydate(restart.variables['time'][0],
                                      units=restart.variables['time'].units,
                                      calendar=restart.variables['time'].calendar)

        if datetime_start != datetime_restart:
            datetime_restart_str = datetime_restart.strftime("%Y-%m-%d %H:%M:%S")
            logger.error(f"The specified start time "
                         f"`{datetime_start_str}' and restart time "
                         f"`{datetime_restart_str}' do not match")
            raise PyLagValueError(f"When restarting the model, the specified "
                                  f"start time should match that given in the "
                                  f"restart file.")

        # Extract particle data
        n_particles = restart.dimensions['particles'].size
        group_ids = restart.variables['group_id'][0, :]

        x1_var_name = variable_library.get_coordinate_variable_name(
                self.coordinate_system, 'x1')
        x1_positions = restart.variables[x1_var_name][0, :]

        x2_var_name = variable_library.get_coordinate_variable_name(
                self.coordinate_system, 'x2')
        x2_positions = restart.variables[x2_var_name][0, :]

        x3_var_name = variable_library.get_coordinate_variable_name(
                self.coordinate_system, 'x3')
        x3_positions = restart.variables[x3_var_name][0, :]

        restart.close()

        return n_particles, group_ids, x1_positions, x2_positions, x3_positions


def get_initial_particle_state_reader(config):
    """ Factor method for particle initial state readers

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuraiton object

    Returns
    -------
     : plag.particle_initialisation.InitialParticleStateReader
        Particle initial state reader

    """
    initialisation_method = config.get("SIMULATION", "initialisation_method")
    if initialisation_method == "init_file":
        return ASCIIInitialParticleStateReader(config)
    elif initialisation_method == "restart_file":
        return RestartInitialParticleStateReader(config)
    elif initialisation_method == "rectangular_grid":
        raise NotImplementedError
    else:
        raise PyLagValueError(f'Unrecognised initialisation method '
                              f'{initialisation_method}')
