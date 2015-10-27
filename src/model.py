import logging

class ParticleTrackingModel(object):
    def __init__(self, config):
        self.config = config
    
    def run(self):
        pass
    
class FVCOMParticleTrackingModel(ParticleTrackingModel):
    def __init__(self, *args, **kwargs):
        super(FVCOMParticleTrackingModel, self).__init__(*args, **kwargs)
        
    def run(self):
        pass

def get_model(config):
    logger = logging.getLogger(__name__)
    if config.get("OCEAN_CIRCULATION_MODEL", "NAME") == "FVCOM":
        logger.info('Creating a new FVCOM particle tracking model.')
        return FVCOMParticleTrackingModel(config)
    else:
        raise ValueError('Unsupported ocean circulation model.')