from particle import get_particle_seed
from netcdf_logger import NetCDFLogger
from model_factory import get_model_factory
from time_manager import TimeManager
    
def run(config):
    # Time manager - for controlling particle releasese event, time stepping etc
    time_manager = TimeManager(config)
    
    # Factory for specific model types (e.g. FVCOM)
    factory = get_model_factory(config)

    # Model specific grid reader (for finding host elements etc)
    grid_reader = factory.make_grid_reader()
    
    # Create seed particle set
    particle_set = get_particle_seed(config)

    # Initialise netCDF data logger
    data_logger = NetCDFLogger(config, len(particle_set))
    
    # TODO - Stuff!
    
    # Close output files
    data_logger.close()