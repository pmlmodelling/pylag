from datetime import datetime

class TimeManager(object):
    def __init__(self, config):
        self.config = config
        
        self._initialise_time_vars()
        
    def _initialise_time_vars(self):
        release_datetime = self.config.get("PARTICLES", "release_datetime")
        self._release_datetime = datetime.strptime(release_datetime, "%Y-%m-%d %H:%M:%S")