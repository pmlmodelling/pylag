from __future__ import print_function

import logging
import traceback
import numpy as np

from mpi4py import MPI

from pylag.time_manager import TimeManager
from pylag.particle_initialisation import get_initial_particle_state_reader
from pylag.restart import RestartFileCreator
from pylag.netcdf_logger import NetCDFLogger
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT

from pylag.parallel.model_factory import get_model


def get_simulator(config):
    if config.get("SIMULATION", "simulation_type") == "trace":
        return TraceSimulator(config)
    else:
        raise ValueError('Unsupported simulation type.')


class Simulator(object):
    def run(self):
        pass


class TraceSimulator(Simulator):
    def __init__(self, config):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Configuration object
        self._config = config

        # Time manager - for controlling time stepping etc
        self.time_manager = TimeManager(self._config)

        # Model object
        self.model = get_model(self._config, self.time_manager.datetime_start, self.time_manager.datetime_end)

        # Initial particle state readers are used to read in intial
        # particle state data. This only happens on the lead process,
        # so we initialise it to None here, then override this below
        # for the root process.
        self.initial_particle_state_reader=None

        # Flag indicating whether or not restart files should be created
        self.create_restarts = self._config.getboolean('RESTART', 'create_restarts') 

        # Restart creators create restart files. This only happens on
        # the lead process, so we initialise it to None here, then
        # override this below for the root process.
        self.restart_creator = None

        # Overrides when on the root process
        if rank == 0:
            self.initial_particle_state_reader = get_initial_particle_state_reader(self._config)
     
            if self.create_restarts:
                self.restart_creator = RestartFileCreator(self._config)

    def run(self):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Read in particle initial positions from file
        if rank == 0:
            # For logging
            logger = logging.getLogger(__name__)

            # Read in particle initial positions from file - these will be used to
            # create the initial particle set.
            try:
                n_particles, group_ids, x_positions, y_positions, z_positions = \
                    self.initial_particle_state_reader.get_particle_data()
            except Exception as e:
                print(traceback.format_exc())
                comm.Abort()

            if n_particles == len(group_ids):
                self.n_particles = n_particles
                logger.info('Particle seed contains {} '\
                    'particles.'.format(self.n_particles))
            else:
                logger.error('Error reading particle initial positions from '\
                    'file. The number of particles specified in the file is '\
                    '{}. The actual number found while parsing the file was '\
                    '{}.'.format(self.n_particles, len(group_ids)))
                comm.Abort()

            # Insist on the even distribution of particles
            if self.n_particles % size == 0:
                my_n_particles = self.n_particles//size
            else:
                logger.error('For now the total number of particles must '\
                    'divide equally among the set of workers. The total '\
                    'number of particles = {}. The total number of workers = '\
                    '{}.'.format(self.n_particles,size))
                comm.Abort()
        else:
            group_ids = None
            x_positions = None
            y_positions = None
            z_positions = None

            my_n_particles = None

        # Broadcast local particle numbers
        my_n_particles = comm.bcast(my_n_particles, root=0)

        # Local arrays for holding particle data
        my_group_ids = np.empty(my_n_particles, dtype=DTYPE_INT)
        my_x_positions = np.empty(my_n_particles, dtype=DTYPE_FLOAT)
        my_y_positions = np.empty(my_n_particles, dtype=DTYPE_FLOAT)
        my_z_positions = np.empty(my_n_particles, dtype=DTYPE_FLOAT)

        # Scatter particles across workers
        comm.Scatter(group_ids,my_group_ids,root=0)
        comm.Scatter(x_positions,my_x_positions,root=0)
        comm.Scatter(y_positions,my_y_positions,root=0)
        comm.Scatter(z_positions,my_z_positions,root=0)

        # Display particle count if running in debug mode
        if self._config.get('GENERAL', 'log_level') == 'DEBUG':
            print('Processor with rank {} is managing {} particles.'.format(rank, my_n_particles))

        # Initialise particle arrays
        self.model.set_particle_data(my_group_ids, my_x_positions, my_y_positions, my_z_positions)

        # Run the ensemble
        run_simulation = True
        while run_simulation:
            # Read data into arrays
            self.model.read_input_data(self.time_manager.time)

            # Seed the model
            self.model.seed(self.time_manager.time)

            if rank == 0:
                # Data logger on the root process
                file_name = ''.join([self._config.get('GENERAL', 'output_file'), '_{}'.format(self.time_manager.current_release)])
                start_datetime = self.time_manager.datetime_start
                self.data_logger = NetCDFLogger(file_name, start_datetime, n_particles)

                # Write particle group ids to file
                self.data_logger.write_group_ids(group_ids)

            # Write initial state to file
            particle_diagnostics = self.model.get_diagnostics(self.time_manager.time)
            self._save_data(particle_diagnostics)

            # The main update loop
            if rank == 0: logger.info('Starting ensemble member {} ...'.format(self.time_manager.current_release))
            while self.time_manager.time < self.time_manager.time_end:
                if rank == 0:
                    percent_complete = self.time_manager.time / self.time_manager.time_end * 100
                    if percent_complete % 10 == 0:
                        logger.info('{}% complete ...'.format(int(percent_complete)))
                try:
                    # Update
                    self.model.update(self.time_manager.time)
                    self.time_manager.update_current_time()

                    # Save diagnostic data
                    if self.time_manager.write_output_to_file() == 1:
                        particle_diagnostics = self.model.get_diagnostics(self.time_manager.time)
                        self._save_data(particle_diagnostics)

                    # Sync diagnostic data to disk
                    if rank == 0 and self.time_manager.sync_data_to_disk() == 1:
                        self.data_logger.sync()

                    # Create restart
                    if self.create_restarts:
                        if self.time_manager.create_restart_file() == 1:
                            particle_data = self.model.get_particle_data()
                            self._create_restart(particle_data)

                    self.model.read_input_data(self.time_manager.time)
                except Exception as e:
                    print(traceback.format_exc())
                    comm.Abort()

            # Close the current data logger
            if rank == 0:
                logger.info('100% complete ...')
                self.data_logger.close()

            # Run another simulation?
            if self.time_manager.new_simulation():
                run_simulation = True

                # Set up data access for the new simulation
                self.model.setup_input_data_access(self.time_manager.datetime_start,
                                                   self.time_manager.datetime_end)
            else:
                run_simulation = False

    def _save_data(self, diags):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        global_diags = {}
        for diag in list(diags.keys()):
            if rank == 0:
                global_diags[diag] = np.empty(self.n_particles, dtype=type(diags[diag][0]))
            else:
                global_diags[diag] = None

        # Pool diagnostics
        for diag in list(diags.keys()):
            comm.Gather(np.array(diags[diag]), global_diags[diag], root=0)

        # Write to file
        if rank == 0:
            self.data_logger.write(self.time_manager.time, global_diags)

    def _create_restart(self, data):
        """ Create restart file

        The real work is done by RestartFileCreator. Here, we simply pool
        particle data from each process before passing it on. Writing
        occurs on the root process only.

        Parameters:
        -----------
        data : dict
            Dictionary containing particle data.

        Returns:
        --------
        N/A
        """

        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        global_data = {}
        for key in list(data.keys()):
            if rank == 0:
                global_data[key] = np.empty(self.n_particles, dtype=type(data[key][0]))
            else:
                global_data[key] = None

        # Pool data
        for key in list(data.keys()):
            comm.Gather(np.array(data[key]), global_data[key], root=0)

        # Write to file
        if rank == 0:
            file_name_stem = 'restart_{}'.format(self.time_manager.current_release)
            datetime_current = self.time_manager.datetime_current
            self.restart_creator.create(file_name_stem, self.n_particles, datetime_current, global_data)

