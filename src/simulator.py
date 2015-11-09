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

        # Create particle seed
        grid_reader.create_particle_set()

        #    # Update variables describing the particle's local environment
        #    # TODO environment is unused here, but will be updated
        #    for idx, particle in enumerate(particle_set):
        #        if particle.in_domain:
        #            environment = grid_reader.get_local_environment(time_manager.time, 
        #                particle.xpos, particle.ypos, particle.zpos,
        #                particle.host_horizontal_elem)

        # Write initial state to file
        grid_reader.write(self.time_manager.time)

        # Close output files
        grid_reader.shutdown()
