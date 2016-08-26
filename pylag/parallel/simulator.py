import logging
import numpy as np

from mpi4py import MPI

from pylag.time_manager import TimeManager
from pylag.particle_positions_reader import read_particle_initial_positions
from pylag.netcdf_logger import NetCDFLogger

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
        # Time manager - for controlling particle release events, time stepping etc
        self.time_manager = TimeManager(config)
    
    def run(self, config):
        # Model object
        self.model = get_model(config)
        
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Read in particle initial positions from file
        if rank == 0:
            # For logging
            logger = logging.getLogger(__name__)
            
            file_name = config.get('SIMULATION', 'initial_positions_file')

            n_particles, group_ids, x_positions, y_positions, z_positions = \
                    read_particle_initial_positions(file_name)

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
                my_n_particles = self.n_particles/size
            else:
                logger.error('For now the total number of particles must '\
                    'divide equally among the set of workers. The total '\
                    'number of particles = {}. The total number of workers = '\
                    '{}.'.format(self.n_particles,size))
                comm.Abort()
                
            # Data logger
            self.data_logger = NetCDFLogger(config, self.n_particles)

            # Write particle group ids to file
            self.data_logger.write_group_ids(group_ids)
        else:
            group_ids = None
            x_positions = None
            y_positions = None
            z_positions = None

            my_n_particles = None

        # Broadcast local particle numbers
        my_n_particles = comm.bcast(my_n_particles, root=0)
        
        # Local arrays for holding particle data
        my_group_ids = np.empty(my_n_particles, dtype=np.int32)
        my_x_positions = np.empty(my_n_particles, dtype=np.float32)
        my_y_positions = np.empty(my_n_particles, dtype=np.float32)
        my_z_positions = np.empty(my_n_particles, dtype=np.float32)

        # Scatter particles across workers
        comm.Scatter(group_ids,my_group_ids,root=0)
        comm.Scatter(x_positions,my_x_positions,root=0)
        comm.Scatter(y_positions,my_y_positions,root=0)
        comm.Scatter(z_positions,my_z_positions,root=0)   

        # Display particle count if running in debug mode
        if config.get('GENERAL', 'log_level') == 'DEBUG':
            print 'Pocessor with rank {} is managing {} particles.'.format(rank, my_n_particles)

        # Initialise time counters, create particle seed
        self.model.initialise(self.time_manager.time, my_group_ids, my_x_positions,
                my_y_positions, my_z_positions)

        # Write initial state to file
        particle_diagnostics = self.model.get_diagnostics(self.time_manager.time)
        self._record(self.time_manager.time, particle_diagnostics)

        # The main update loop
        #while self.time_manager.time < self.time_manager.time_end:
        #    model.update(self.time_manager.time)
        #    self.time_manager.update_current_time()
        #    if self.time_manager.write_output_to_file() == 1:
        #        model.record(self.time_manager.time)
        #    model.update_reading_frame(self.time_manager.time)

        # Close output files
        #model.shutdown()

    def _record(self, time, diags):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        global_diags = {}
        for diag in diags.keys():
            if rank == 0:
                global_diags[diag] = np.empty(self.n_particles, dtype=type(diags[diag][0]))
            else:
                global_diags[diag] = None
        
        # Pool diagnostics
        for diag in diags.keys():
            comm.Gather(np.array(diags[diag]), global_diags[diag], root=0)

        # Write to file
        if rank == 0:
            self.data_logger.write(time, global_diags)
