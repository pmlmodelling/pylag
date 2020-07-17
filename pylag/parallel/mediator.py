"""
Module containing the derived class MPIMediator, which helps to manage
access to input data during parallel execution.

See Also
--------
pylag.mediator - Serial mediator for serial execution
"""

import numpy as np
import logging
import traceback

# For parallel simulations
from mpi4py import MPI

from pylag.data_types_python import DTYPE_INT
from pylag.file_reader import FileReader
from pylag.file_reader import DiskFileNameReader
from pylag.file_reader import NetCDFDatasetReader
from pylag.mediator import Mediator


class MPIMediator(Mediator):
    """ MPI mediator

    MPI mediator for parallel runs.

    Parameters
    ----------
    config : ConfigParser
        Run configuration object

    start_datetime : Datetime
        Simulation start date/time.

    end_datetime : Datetime
        Simulation end date/time.

    Attributes
    ----------
    config : ConfigParser
        Run configuration object

    file_reader : pylag.FileReader
        FileReader object.

    """
    def __init__(self, config, datetime_start, datetime_end):
        self.config = config

        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # Only the root process accesses the file system
        if rank == 0:
            try:
                file_name_reader = DiskFileNameReader()
                dataset_reader = NetCDFDatasetReader()
                self.file_reader = FileReader(config, file_name_reader, dataset_reader, datetime_start, datetime_end)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when reading input file. '\
                'Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()
        else:
            self.file_reader = None

    def setup_data_access(self, datetime_start, datetime_end):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            try:
                self.file_reader.setup_data_access(datetime_start, datetime_end)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when setting up data access. '\
                'Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()

    def update_reading_frames(self, time):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            try:
                self.file_reader.update_reading_frames(time)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when updating reading frames. '\
                'Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()

    def get_dimension_variable(self, var_name):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            try:
                var = self.file_reader.get_dimension_variable(var_name)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when getting dimension variable. '\
                'Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()
        else:
            var = None

        var = comm.bcast(var, root=0)
        
        return var

    def get_grid_variable(self, var_name, var_dims, var_type):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            try:
                var = self.file_reader.get_grid_variable(var_name).astype(var_type)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when getting grid variable. '\
                'Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()
        else:
            var = np.empty(var_dims, dtype=var_type)
        
        comm.Bcast(var, root=0)
        
        return var

    def get_time_at_last_time_index(self):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            try:
                time = self.file_reader.get_time_at_last_time_index()
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when getting last time index. '\
                'Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()
        else:
            time = None
        
        time = comm.bcast(time, root=0)
        
        return time

    def get_time_at_next_time_index(self):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            try:
                time = self.file_reader.get_time_at_next_time_index()
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when getting next time index. '\
                'Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()
        else:
            time = None
        
        time = comm.bcast(time, root=0)
        
        return time

    def get_grid_variable_dimensions(self, var_name):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            try:
                dimensions = self.file_reader.get_grid_variable_dimensions(var_name)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when getting variable dimensions. ' \
                             'Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()
        else:
            dimensions = None

        dimensions = comm.bcast(dimensions, root=0)

        return dimensions

    def get_variable_dimensions(self, var_name):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            try:
                dimensions = self.file_reader.get_variable_dimensions(var_name)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when getting variable dimensions. ' \
                             'Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()
        else:
            dimensions = None

        dimensions = comm.bcast(dimensions, root=0)

        return dimensions

    def get_variable_shape(self, var_name):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            try:
                shape = self.file_reader.get_variable_shape(var_name)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when getting variable shape. ' \
                             'Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()
        else:
            shape = None

        shape = comm.bcast(shape, root=0)

        return shape

    def get_time_dependent_variable_at_last_time_index(self, var_name, var_dims, var_type):
         # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            try:
                var = self.file_reader.get_time_dependent_variable_at_last_time_index(var_name).astype(var_type)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when getting time variable at '\
                'last time index. Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()
        else:
            var = np.empty(var_dims, dtype=var_type)
        
        comm.Bcast(var, root=0)
        
        return var

    def get_time_dependent_variable_at_next_time_index(self, var_name, var_dims, var_type):
         # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            try:
                var = self.file_reader.get_time_dependent_variable_at_next_time_index(var_name).astype(var_type)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when getting time variable at '\
                'next time index. Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()
        else:
            var = np.empty(var_dims, dtype=var_type)
        
        comm.Bcast(var, root=0)
        
        return var

    def get_mask_at_last_time_index(self, var_name, var_dims):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            try:
                mask = self.file_reader.get_mask_at_last_time_index(var_name).astype(DTYPE_INT)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when getting mask at ' \
                             'last time index. Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()
        else:
            mask = np.empty(var_dims, dtype=DTYPE_INT)

        comm.Bcast(mask, root=0)

        return mask

    def get_mask_at_next_time_index(self, var_name, var_dims):
        # MPI objects and variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            try:
                mask = self.file_reader.get_mask_at_next_time_index(var_name).astype(DTYPE_INT)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error('Caught exception when getting mask at ' \
                             'next time index. Terminating all tasks ...')
                logger.error(traceback.format_exc())
                comm.Abort()
        else:
            mask = np.empty(var_dims, dtype=DTYPE_INT)

        comm.Bcast(mask, root=0)

        return mask

