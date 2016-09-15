import numpy as np

# For parallel simulations
from mpi4py import MPI

from pylag.file_reader import FileReader
from pylag.mediator import Mediator

class MPIMediator(Mediator):
    def __init__(self, config):
        self.config = config

        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # Only the root process accesses the file system
        if rank == 0:
            self.file_reader = FileReader(config)
        else:
            self.file_reader = None

    def setup_data_access(self, start_datetime, end_datetime):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            self.file_reader.setup_data_access(start_datetime, end_datetime)

    def update_reading_frames(self, time):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            self.file_reader.update_reading_frames(time)

    def get_dimension_variable(self, var_name):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            var = self.file_reader.get_dimension_variable(var_name)
        else:
            var = None

        var = comm.bcast(var, root=0)
        
        return var

    def get_grid_variable(self, var_name, var_dims=None, var_type=None):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            var = self.file_reader.get_grid_variable(var_name)
        else:
            var = np.empty(var_dims, dtype=var_type)
        
        comm.Bcast(var, root=0)
        
        return var

    def get_time_at_last_time_index(self):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            time = self.file_reader.get_time_at_last_time_index()
        else:
            time = None
        
        time = comm.bcast(time, root=0)
        
        return time

    def get_time_at_next_time_index(self):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            time = self.file_reader.get_time_at_next_time_index()
        else:
            time = None
        
        time = comm.bcast(time, root=0)
        
        return time
    
    def get_time_dependent_variable_at_last_time_index(self, var_name, var_dims=None, var_type=None):
         # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            var = self.file_reader.get_time_dependent_variable_at_last_time_index(var_name)
        else:
            var = np.empty(var_dims, dtype=var_type)
        
        comm.Bcast(var, root=0)
        
        return var

    def get_time_dependent_variable_at_next_time_index(self, var_name, var_dims=None, var_type=None):
         # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            var = self.file_reader.get_time_dependent_variable_at_next_time_index(var_name)
        else:
            var = np.empty(var_dims, dtype=var_type)
        
        comm.Bcast(var, root=0)
        
        return var
    