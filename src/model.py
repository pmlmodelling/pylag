from particle import get_particle_seed
from netcdf_logger import NetCDFLogger

class ModelFactory(object):
    def __init__(self, config):
        self.config = config
    
    def make_grid_reader(self): pass

class FVCOMModelFactory(ModelFactory):
    def __init__(self, *args, **kwargs):
        super(FVCOMModelFactory, self).__init__(*args, **kwargs)

    def make_grid_reader(self):
        pass

def get_model_factory(config):
    if config.get("OCEAN_CIRCULATION_MODEL", "NAME") == "FVCOM":
        return FVCOMModelFactory(config)
    else:
        raise ValueError('Unsupported ocean circulation model.')    
    
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