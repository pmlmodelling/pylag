import logging
from datetime import datetime

import model_reader
from particle import get_particle_seed
from netcdf_logger import NetCDFLogger

class OPTModel(object):
    def __init__(self, config):
        self.config = config

    def initialise(self, time):
        pass
    
    def advect(self, time, time_step):
        pass
    
    def rand_walk(self):
        pass
    
    def record(self, time):
        pass
    
    def shutdown(self):
        pass
    
class FVCOMOPTModel(OPTModel):
    def __init__(self, *args, **kwargs):
        super(FVCOMOPTModel, self).__init__(*args, **kwargs)

        self.model_reader = model_reader.FVCOMModelReader(self.config.get("OCEAN_CIRCULATION_MODEL", "data_file"),
                                                          self.config.get("OCEAN_CIRCULATION_MODEL", "grid_metrics_file"))

    def initialise(self, time):
        # Index corresponding to the last time point at which time dependent
        # model variables are defined
        self._tidx_last = None
        
        # Index corresponding to the next time point at which time dependent 
        # model variables are defined
        self._tidx_next = None

        # Float giving the fractional position between the last and next time
        # steps
        self._time_fraction = None

        # Initialise vars
        self._time_fraction, self._tidx_last, self._tidx_next = self.model_reader.get_time(time)

        # Create seed particle set
        self._create_particle_set()

    def advect(self, time, time_step):
        pass
        
    def record(self, time):
        # Intialise data logger
        if not hasattr(self, "data_logger"):
            self.data_logger = NetCDFLogger(self.config, len(self.particle_set))

        # Write particle initial positions to file
        self.data_logger.write(time, self.particle_set)
        
    def shutdown(self):
        self.data_logger.close()

    def _create_particle_set(self):
        # Create seed particle set
        self.particle_set = get_particle_seed(self.config)

        # Find particle host elements within the model domain and initalise the
        # particle's local environment
        guess = None
        particles_in_domain = 0
        for idx, particle in enumerate(self.particle_set):
            self.particle_set[idx].host_horizontal_elem = self.model_reader.find_host(particle.xpos, particle.ypos, guess)

            if self.particle_set[idx].host_horizontal_elem != -1:
                self.particle_set[idx].in_domain = True

                self.particle_set[idx].h = self.model_reader.get_bathymetry(particle.xpos, 
                        particle.ypos, particle.host_horizontal_elem)

                self.particle_set[idx].zeta = self.model_reader.get_sea_sur_elev(self._tidx_last,
                        self._tidx_next, self._time_fraction, particle.xpos, 
                        particle.ypos, particle.host_horizontal_elem)

                particles_in_domain += 1

                # Use the location of the last particle to guide the search for the
                # next. This should be fast if particle initial positions are colocated.
                guess = self.particle_set[idx].host_horizontal_elem
            else:
                self.particle_set[idx].in_domain == False

        logger = logging.getLogger(__name__)
        logger.info('{} of {} particles are located in the model domain.'.format(particles_in_domain, len(self.particle_set)))

def get_model(config):
    if config.get("OCEAN_CIRCULATION_MODEL", "name") == "FVCOM":
        return FVCOMOPTModel(config)
    else:
        raise ValueError('Unsupported ocean circulation model.')    
