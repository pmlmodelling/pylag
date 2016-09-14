from pylag.file_reader import FVCOMFileReader

class Mediator(object):
    """ Base class for objects of type Mediator.

    Mediators manage data transfers between FileReaders and other objects. Data
    transfers are typically between objects of type FileReader and objects of 
    type DataReader. The decoupling between FileReader and DataReader objects
    was put in place in order to support parallel simulations where it is 
    preferable for just a single process to access the file system.
    
    """
    def set_time_counters(self, start_datetime, end_datetime):
        pass

    def update_reading_frames(self, time):
        pass
    
    def get_dimension_variable(self, var_name):
        pass
    
    def get_grid_variable(self, var_name, var_dims=None, var_type=None):
        pass

    def get_time_at_last_time_index(self):
        pass

    def get_time_at_next_time_index(self):
        pass

    def get_time_dependent_variable_at_last_time_index(self, var_name, var_dims=None, var_type=None):
        pass

    def get_time_dependent_variable_at_next_time_index(self, var_name, var_dims=None, var_type=None):
        pass

class FVCOMMediator(Mediator):
    def __init__(self, config):
        self.config = config
        self.file_reader = FVCOMFileReader(config)

    def set_time_counters(self, start_datetime, end_datetime):
        self.file_reader.set_time_counters(start_datetime, end_datetime)

    def update_reading_frames(self, time):
        return self.file_reader.update_reading_frames(time)

    def get_dimension_variable(self, var_name):
        return self.file_reader.get_dimension_variable(var_name)
    
    def get_grid_variable(self, var_name, var_dims=None, var_type=None):
        return self.file_reader.get_grid_variable(var_name)

    def get_time_at_last_time_index(self):
        return self.file_reader.get_time_at_last_time_index()

    def get_time_at_next_time_index(self):
        return self.file_reader.get_time_at_next_time_index()

    def get_time_dependent_variable_at_last_time_index(self, var_name, var_dims=None, var_type=None):
        return self.file_reader.get_time_dependent_variable_at_last_time_index(var_name)

    def get_time_dependent_variable_at_next_time_index(self, var_name, var_dims=None, var_type=None):
        return self.file_reader.get_time_dependent_variable_at_next_time_index(var_name)
