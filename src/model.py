import logging

from particle import get_particle_seed
from netcdf_logger import NetCDFLogger
from model_factory import get_model_factory
from time_manager import TimeManager
    
def run(config):
    # Factory for specific model types (e.g. FVCOM)
    factory = get_model_factory(config)

    # Model specific grid reader (for finding host elements etc)
    grid_reader = factory.make_grid_reader()

    # Time manager - for controlling particle release events, time stepping etc
    time_manager = TimeManager(config)

    # Create seed particle set
    particle_set = get_particle_seed(config)

    # Find particle host elements within the model domain
    guess = 0
    found_using_local_search = 0
    found_using_global_search = 0
    for idx, particle in enumerate(particle_set):
        try:
            particle_set[idx].host_horizontal_elem = grid_reader.find_host_using_local_search(particle, guess)
            particle_set[idx].in_domain = True
            found_using_local_search += 1
        except ValueError:
            try:
                # Local search failed - try global search
                particle_set[idx].host_horizontal_elem = grid_reader.find_host_using_global_search(particle)
                particle_set[idx].in_domain = True
                found_using_global_search += 1
            except ValueError:
                # Gloabl search failed - flag the particle as being outside the domain
                particle_set[idx].in_domain = False

        # Use the location of the last particle to guide the search for the
        # next. This should be fast if particle initial positions are colocated.
        if particle_set[idx].in_domain == True:
            guess = particle_set[idx].host_horizontal_elem

    # Report on the number of particles found in the domain
    particles_in_domain = 0
    for particle in particle_set:
        if particle.in_domain == True:
            particles_in_domain += 1
    logger = logging.getLogger(__name__)
    logger.info('{} of {} particles are located in the model domain.'.format(particles_in_domain, len(particle_set)))
    logger.info('{} were found using local searching, {} using global searching.'.format(found_using_local_search, found_using_global_search))

    # Initialise netCDF data logger
    data_logger = NetCDFLogger(config, len(particle_set))
    
    # Write particle initial positions to file
    data_logger.write(time_manager.time, particle_set)

    # Close output files
    data_logger.close()