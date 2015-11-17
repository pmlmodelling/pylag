import logging

from fvcom_data_reader import FVCOMDataReader
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

    def initialise(self, time):
        # Create FVCOM data reader
        self.data_reader = FVCOMDataReader(self.config)

        # Create seed particle set
        self.particle_set = get_particle_seed(self.config)

        # Find particle host elements within the model domain and initalise the
        # particle's local environment
        guess = None
        particles_in_domain = 0
        for idx, particle in enumerate(self.particle_set):
            self.particle_set[idx].host_horizontal_elem = self.data_reader.find_host(particle.xpos, particle.ypos, guess)

            if self.particle_set[idx].host_horizontal_elem != -1:
                self.particle_set[idx].in_domain = True

                self.particle_set[idx].h = self.data_reader.get_bathymetry(particle.xpos, 
                        particle.ypos, particle.host_horizontal_elem)

                self.particle_set[idx].zeta = self.data_reader.get_sea_sur_elev(time, particle.xpos, 
                        particle.ypos, particle.host_horizontal_elem)

                particles_in_domain += 1

                # Use the location of the last particle to guide the search for the
                # next. This should be fast if particle initial positions are colocated.
                guess = self.particle_set[idx].host_horizontal_elem
            else:
                self.particle_set[idx].in_domain == False

        logger = logging.getLogger(__name__)
        logger.info('{} of {} particles are located in the model domain.'.format(particles_in_domain, len(self.particle_set)))

    def advect(self, time, time_step):
        pass
        
    def record(self, time):
        # Intialise data logger
        if not hasattr(self, "data_logger"):
            self.data_logger = NetCDFLogger(self.config, len(self.particle_set))

        # Write particle data to file
        particle_data = create_lists(self.particle_set)
        self.data_logger.write(time, particle_data)
        
    def shutdown(self):
        self.data_logger.close()

def get_model(config):
    if config.get("OCEAN_CIRCULATION_MODEL", "name") == "FVCOM":
        return FVCOMOPTModel(config)
    else:
        raise ValueError('Unsupported ocean circulation model.')
    
def create_lists(particle_array):
    data = {'xpos': [], 'ypos': [], 'zpos': [], 'h': [], 'zeta': []}
    for particle in particle_array:
        data['xpos'].append(particle.xpos)
        data['ypos'].append(particle.ypos)
        data['zpos'].append(particle.zpos)
        data['h'].append(particle.h)
        data['zeta'].append(particle.zeta)
    return data