import logging

from particle import get_particle_seed
from netcdf_logger import NetCDFLogger
from model_factory import get_model_factory
from time_manager import TimeManager

def get_simulator(config):
    if config.get("GENERAL", "simulation_type") == "trace":
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
        
        # Factory for specific model types (e.g. FVCOM)
        factory = get_model_factory(config)

        # Model specific grid reader (for finding host elements etc)
        grid_reader = factory.make_grid_reader()

        # Create seed particle set
        particle_set = get_particle_seed(config)

        # Find particle host elements within the model domain
        guess = None
        particles_in_domain = 0
        for idx, particle in enumerate(particle_set):
            particle_set[idx].host_horizontal_elem = grid_reader.find_host(particle.xpos, particle.ypos, guess)

            if particle_set[idx].host_horizontal_elem != -1:
                particle_set[idx].in_domain = True
                particles_in_domain += 1

                # Use the location of the last particle to guide the search for the
                # next. This should be fast if particle initial positions are colocated.
                guess = particle_set[idx].host_horizontal_elem
            else:
                particle_set[idx].in_domain == False

        logger = logging.getLogger(__name__)
        logger.info('{} of {} particles are located in the model domain.'.format(particles_in_domain, len(particle_set)))

        #    # Update variables describing the particle's local environment
        #    # TODO environment is unused here, but will be updated
        #    for idx, particle in enumerate(particle_set):
        #        if particle.in_domain:
        #            environment = grid_reader.get_local_environment(time_manager.time, 
        #                particle.xpos, particle.ypos, particle.zpos,
        #                particle.host_horizontal_elem)

        # Initialise netCDF data logger
        data_logger = NetCDFLogger(config, len(particle_set))

        # Write particle initial positions to file
        data_logger.write(self.time_manager.time, particle_set)

        # Close output files
        data_logger.close()
