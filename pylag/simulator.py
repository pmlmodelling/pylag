import logging
from progressbar import ProgressBar

from pylag.model import get_model
from pylag.time_manager import TimeManager
from pylag.particle_positions_reader import read_particle_initial_positions

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
        model = get_model(config)

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
        model.initialise(self.time_manager.time, group_ids, x_positions, \
            y_positions, z_positions)

        # Write initial state to file
        model.record(self.time_manager.time)

        # The main update loop
        print 'Starting PyLag\n'
        print 'Progress:'
        pbar = ProgressBar(maxval=self.time_manager.time_end, term_width=50).start()
        while self.time_manager.time < self.time_manager.time_end:
            model.update(self.time_manager.time)
            self.time_manager.update_current_time()
            if self.time_manager.write_output_to_file() == 1:
                model.record(self.time_manager.time)
            model.update_reading_frame(self.time_manager.time)
            pbar.update(self.time_manager.time)
        pbar.finish()

        # Close output files
        model.shutdown()
