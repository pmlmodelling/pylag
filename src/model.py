import logging

import model_reader
from particle import get_particle_seed
from netcdf_logger import NetCDFLogger

class OPTModel(object):
    def __init__(self, config):
        self.config = config

    def create_particle_set(self):
        pass
    
    def write(self, time):
        pass
    
    def shutdown(self):
        pass
    
class FVCOMOPTModel(OPTModel):
    def __init__(self, *args, **kwargs):
        super(FVCOMOPTModel, self).__init__(*args, **kwargs)

        self.model_reader = model_reader.FVCOMModelReader(self.config.get("OCEAN_CIRCULATION_MODEL", "data_file"),
                                                          self.config.get("OCEAN_CIRCULATION_MODEL", "grid_metrics_file"))

    def create_particle_set(self):
        # Create seed particle set
        self.particle_set = get_particle_seed(self.config)

        # Find particle host elements within the model domain
        guess = None
        particles_in_domain = 0
        for idx, particle in enumerate(self.particle_set):
            self.particle_set[idx].host_horizontal_elem = self.model_reader.find_host(particle.xpos, particle.ypos, guess)

            if self.particle_set[idx].host_horizontal_elem != -1:
                self.particle_set[idx].in_domain = True
                particles_in_domain += 1

                # Use the location of the last particle to guide the search for the
                # next. This should be fast if particle initial positions are colocated.
                guess = self.particle_set[idx].host_horizontal_elem
            else:
                self.particle_set[idx].in_domain == False

        logger = logging.getLogger(__name__)
        logger.info('{} of {} particles are located in the model domain.'.format(particles_in_domain, len(self.particle_set)))

    def write(self, time):
        # Intialise data logger
        if not hasattr(self, "data_logger"):
            self.data_logger = NetCDFLogger(self.config, len(self.particle_set))

        # Write particle initial positions to file
        self.data_logger.write(time, self.particle_set)        
        
    def shutdown(self):
        self.data_logger.close()

def get_model(config):
    if config.get("OCEAN_CIRCULATION_MODEL", "name") == "FVCOM":
        return FVCOMOPTModel(config)
    else:
        raise ValueError('Unsupported ocean circulation model.')    
