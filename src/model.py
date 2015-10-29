import logging

from particle import get_particle_seed
from netcdf_logger import NetCDFLogger

class ParticleTrackingModel(object):
    def __init__(self, config):
        self.config = config
    
    def run(self):
        pass
    
    def shutdown(self):
        pass
    
class FVCOMParticleTrackingModel(ParticleTrackingModel):
    def __init__(self, *args, **kwargs):
        super(FVCOMParticleTrackingModel, self).__init__(*args, **kwargs)
        
    def run(self):
        # Create seed particle set
        self.particle_set = get_particle_seed(self.config)
        
        # Initialise netCDF data logger
        self.data_logger = NetCDFLogger(self.config, len(self.particle_set))
        
    def shutdown(self):
        if hasattr(self, 'data_logger'):
            self.data_logger.close()

def get_model(config):
    logger = logging.getLogger(__name__)
    if config.get("OCEAN_CIRCULATION_MODEL", "NAME") == "FVCOM":
        logger.info('Creating a new FVCOM particle tracking model.')
        return FVCOMParticleTrackingModel(config)
    else:
        raise ValueError('Unsupported ocean circulation model.')