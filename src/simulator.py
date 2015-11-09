from model_factory import get_model
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
        # Model object
        model = get_model(config)

        # Create particle seed
        model.create_particle_set()

        #    # Update variables describing the particle's local environment
        #    # TODO environment is unused here, but will be updated
        #    for idx, particle in enumerate(particle_set):
        #        if particle.in_domain:
        #            environment = grid_reader.get_local_environment(time_manager.time, 
        #                particle.xpos, particle.ypos, particle.zpos,
        #                particle.host_horizontal_elem)

        # Write initial state to file
        model.write(self.time_manager.time)

        # Close output files
        model.shutdown()
