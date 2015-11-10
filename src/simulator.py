from model import get_model
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

        # Write initial state to file
        model.record(self.time_manager.time)

        # Close output files
        model.shutdown()
