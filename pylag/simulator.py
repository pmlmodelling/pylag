"""
This module contains classes that can be used to manage the running of
PyLag simulations in serial mode. All Simulators inherit from a common
Abstract base class `Simulator`. This structure was introduced in order
to make it possible to run different types of simulation (e.g. a trace,
or LCS calculation) through configuration switches.

See Also
--------
pylag.parallel.simulator - Simulators for parallel execution
"""

from __future__ import print_function

import logging
from progressbar import ProgressBar

from pylag.time_manager import TimeManager
from pylag.particle_initialisation import get_initial_particle_state_reader
from pylag.restart import RestartFileCreator
from pylag.netcdf_logger import NetCDFLogger
from pylag.exceptions import PyLagValueError, PyLagRuntimeError

from pylag.model_factory import get_model


def get_simulator(config):
    """ Factory method for PyLag simulators

    Parameters
    ----------
    config : ConfigParser
        PyLag configuraton object

    Returns
    -------
     : pylag.simulator.Simulator
         Object of type Simulator
    """
    if config.get("SIMULATION", "simulation_type") == "trace":
        return TraceSimulator(config)
    else:
        raise PyLagValueError('Unsupported simulation type.')


class Simulator(object):
    """ Simulator

    Abstract base class for PyLag simulators.
    """
    def run(self):
        """ Run a PyLag simulation
        """
        raise NotImplementedError


class TraceSimulator(Simulator):
    """ Trace simulator

    Simulator for tracing particle pathlines through time. Trace simulators
    can perform forward or backward in time integrations.

    Parameters
    ----------
    config : ConfigParser
        PyLag configuraton object
    """
    def __init__(self, config):
        # Configuration object
        self._config = config

        # Time manager - for controlling particle release events, time stepping
        self.time_manager = TimeManager(self._config)
    
        # Model object
        self.model = get_model(self._config, self.time_manager.datetime_start,
                               self.time_manager.datetime_end)

        # Initial particle state reader
        self.initial_particle_state_reader = get_initial_particle_state_reader(config)

        # Restart creator
        self.restart_creator = None
        if self._config.getboolean('RESTART', 'create_restarts'):
            self.restart_creator = RestartFileCreator(config)

        # Data logger
        self.data_logger = None
    
    def run(self):
        """ Run a simulation

        Run a single or multiple integrations according to options set out
        in the run configuration file.

        Returns
        -------
         : None
        """
        # For logging
        logger = logging.getLogger(__name__)

        # Read in particle initial positions from file - these will be used to
        # create the initial particle set.
        n_particles, group_ids, x1_positions, x2_positions, x3_positions = \
            self.initial_particle_state_reader.get_particle_data()

        if n_particles == len(group_ids):
            logger.info(f'Particle seed contains {n_particles} particles.')
        else:
            logger.error(f'Error reading particle initial positions from '
                         f'file. The number of particles specified in the file '
                         f'is {n_particles}. The actual number found while '
                         f'parsing the file was {len(group_ids)}.')

            raise PyLagRuntimeError('Error encountered while reading the '
                                    'particle initial positions file. See '
                                    'the log for more details.')

        # Initialise particle arrays
        self.model.set_particle_data(group_ids, x1_positions, x2_positions,
                                     x3_positions)

        # Run the ensemble
        run_simulation = True
        while run_simulation:
            # Read data into arrays
            self.model.read_input_data(self.time_manager.time)
            
            # Seed the model
            self.model.seed(self.time_manager.time)
            
            # Create data logger
            file_name = ''.join([self._config.get('GENERAL', 'output_file'),
                                 f'_{self.time_manager.current_release}'])
            start_datetime = self.time_manager.datetime_start
            grid_names = self.model.get_grid_names()
            self.data_logger = NetCDFLogger(self._config, file_name,
                                            start_datetime, n_particles,
                                            grid_names)

            # Write particle group ids to file
            self.data_logger.write_group_ids(group_ids)

            # Write initial state to file
            particle_diagnostics = self.model.get_diagnostics(self.time_manager.time)
            self.data_logger.write(self.time_manager.time, particle_diagnostics)

            # The main update loop
            print(f'\nStarting ensemble member '
                  f'{self.time_manager.current_release} ...')
            print('Progress:')
            pbar = ProgressBar(maxval=abs(self.time_manager.time_end),
                               term_width=50).start()
            while abs(self.time_manager.time) < abs(self.time_manager.time_end):
                self.model.update(self.time_manager.time)
                self.time_manager.update_current_time()

                # Check on status of reading frames and update if necessary
                # Communicate updated arrays to the data reader if these are
                # out of date.
                self.model.read_input_data(self.time_manager.time)

                if self.time_manager.write_output_to_file() == 1:
                    particle_diagnostics = self.model.get_diagnostics(
                            self.time_manager.time)
                    self.data_logger.write(self.time_manager.time,
                                           particle_diagnostics)
                
                if self.time_manager.sync_data_to_disk() == 1:
                    self.data_logger.sync()

                if self.restart_creator:
                    if self.time_manager.create_restart_file() == 1:
                        file_name_stem = f'restart_{self.time_manager.current_release}'
                        datetime_current = self.time_manager.datetime_current
                        particle_data = self.model.get_particle_data()

                        self.restart_creator.create(file_name_stem, n_particles,
                                                    datetime_current,
                                                    particle_data)

                pbar.update(abs(self.time_manager.time))
            pbar.finish()
            
            # Close the current data logger
            self.data_logger.close()

            # Run another simulation?
            if self.time_manager.new_simulation():
                run_simulation = True

                # Set up data access for the new simulation
                self.model.setup_input_data_access(
                        self.time_manager.datetime_start,
                        self.time_manager.datetime_end)
            else:
                run_simulation = False


__all__ = ['Simulator',
           'TraceSimulator',
           'get_simulator']
