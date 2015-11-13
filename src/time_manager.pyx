import copy
import datetime

class TimeManager(object):
    def __init__(self, config):
        self.config = config
        
        self._initialise_time_vars()
        
    def _initialise_time_vars(self):
        start_datetime = self.config.get("PARTICLES", "start_datetime")
        self._time_start = datetime.datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")

        end_datetime = self.config.get("PARTICLES", "end_datetime")
        self._time_end = datetime.datetime.strptime(end_datetime, "%Y-%m-%d %H:%M:%S")
        
        self._time_step = self.config.getint("PARTICLES", "time_step")
        
        self._output_frequency = self.config.getint("PARTICLES", "output_frequency")

        # Set the current time to the start time
        self._time = copy.deepcopy(self._time_start)

    def update_current_time(self):
        self._time = self._time + datetime.timedelta(0, self._time_step)
        
    def write_output_to_file(self):
        time_diff = (self._time - self._time_start).total_seconds()
        if time_diff % self._output_frequency == 0:
            return True
        return False
        
    # Current date and time
    @property
    def time(self):
        return self._time
    
    @time.setter
    def time(self, value):
        self._time = value

    # Integration time step
    @property
    def time_step(self):
        return self._time_step
    
    @time_step.setter
    def time_step(self, value):
        self._time_step = value

    # Integration end time
    @property
    def time_end(self):
        return self._time_end
    
    @time_end.setter
    def time_end(self, value):
        self._time_end = value
