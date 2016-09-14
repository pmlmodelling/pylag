import logging
from progressbar import ProgressBar

from pylag.time_manager import TimeManager
from pylag.particle_positions_reader import read_particle_initial_positions
from pylag.netcdf_logger import NetCDFLogger

from pylag.model_factory import get_model

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
        # For logging
        logger = logging.getLogger(__name__)
        
        # Model object
        self.model = get_model(config)

        # Set up data access for the simulation start and end times
        self.model.read_data(self.time_manager.datetime_start,
                             self.time_manager.datetime_end)

        # Read in particle initial positions from file - these will be used to
        # create the initial particle set.
        file_name = config.get('SIMULATION', 'initial_positions_file')
        n_particles, group_ids, x_positions, y_positions, z_positions = \
            read_particle_initial_positions(file_name)

        if n_particles == len(group_ids):
            logger.info('Particle seed contains {} '\
                'particles.'.format(n_particles))
        else:
            logger.error('Error reading particle initial positions from '\
                'file. The number of particles specified in the file is '\
                '{}. The actual number found while parsing the file was '\
                '{}.'.format(n_particles, len(group_ids)))

            raise RuntimeError('Error encountered while reading the particle '\
                'initial positions file {}.'.format(file_name))

        # Initialise time counters, create particle seed
        self.model.create_seed(self.time_manager.time, group_ids, x_positions, \
            y_positions, z_positions)

        # Run the ensemble
        release = 0
        while release < self.time_manager.number_of_particle_releases:
            # Seed the model
            self.model.seed()
            
            # Create data logger
            file_name = ''.join([config.get('GENERAL', 'output_file'), '_{}'.format(release)])
            start_datetime = self.time_manager.datetime_start
            self.data_logger = NetCDFLogger(file_name, start_datetime, n_particles)

            # Write particle group ids to file
            self.data_logger.write_group_ids(group_ids)

            # Write initial state to file
            particle_diagnostics = self.model.get_diagnostics(self.time_manager.time)
            self.data_logger.write(self.time_manager.time, particle_diagnostics)

            # The main update loop
            print '\nStarting ensemble member {} ...'.format(release)
            print 'Progress:'
            pbar = ProgressBar(maxval=self.time_manager.time_end, term_width=50).start()
            while self.time_manager.time < self.time_manager.time_end:
                self.model.update(self.time_manager.time)
                self.time_manager.update_current_time()
                if self.time_manager.write_output_to_file() == 1:
                    particle_diagnostics = self.model.get_diagnostics(self.time_manager.time)
                    self.data_logger.write(self.time_manager.time, particle_diagnostics)

                # Check on status of reading frames and update if necessary
                # Communicate updated arrays to the data reader if these are out of
                # date.
                self.model.update_reading_frames(self.time_manager.time)
                pbar.update(self.time_manager.time)
            pbar.finish()
            
            # Close the current data logger
            self.data_logger.close()
            
            # Update release counters
            self.time_manager.update_release_counters()

            # Set up data access for the new simulation
            self.model.read_data(self.time_manager.datetime_start,
                                 self.time_manager.datetime_end)

            # Update release counter
            release += 1
