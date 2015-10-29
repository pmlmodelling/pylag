from particle import get_particle_seed
from netcdf_logger import NetCDFLogger
from model_factory import get_model_factory
    
def run(config):
    factory = get_model_factory(config)

    grid_reader = factory.make_grid_reader()
    
    # Create seed particle set
    particle_set = get_particle_seed(config)

    # Initialise netCDF data logger
    data_logger = NetCDFLogger(config, len(particle_set))
    
    # TODO - Stuff!
    
    # Close output files
    data_logger.close()