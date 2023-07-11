"""
The module provides a set of classes to assist with communication between
objects of type DataReader and objects of type FileReader. Primarily, the
mediator module was introduced in order to abstract away the MPI interface,
making it easier to PyLag either in serial or parallel. It also assists with
testing.

See Also
--------
pylag.parallel.mediator - MPI mediator for parallel execution
"""

from pylag.data_types_python import DTYPE_INT
from pylag.file_reader import FileReader
from pylag.file_reader import DiskFileNameReader
from pylag.file_reader import NetCDFDatasetReader


class Mediator:
    """ Base class for objects of type Mediator.

    Mediators manage data transfers between FileReaders and other objects. Data
    transfers are typically between objects of type FileReader and objects of 
    type DataReader. The decoupling between FileReader and DataReader objects
    was put in place in order to support parallel simulations where it is 
    preferable for just a single process to access the file system.
    """
    def setup_data_access(self, start_datetime, end_datetime):
        """Wrapper for FileReader's setup_data_access

        Parameters
        ----------
        start_datetime : Datetime
            Simulation start date/time.

        end_datetime : Datetime
            Simulation end date/time.
        """
        raise NotImplementedError

    def update_reading_frames(self, time):
        """ Wrapper for FileReader's update_reading_frames

        Parameters
        ----------
        time : float
            Time
        """
        raise NotImplementedError
    
    def get_dimension_variable(self, var_name):
        """ Wrapper for FileReader's get_dimension_variable

        Parameters
        ----------
        var_name : str
            The name of the dimension variable.

        Returns
        -------
         : int
             The size of the dimensions variable.
        """
        raise NotImplementedError
    
    def get_grid_variable(self, var_name, var_dims, var_type):
        """ Wrapper for FileReader's get_grid_variable

        Parameters
        ----------
        var_name : str
            The name of the variable.

        Returns
        -------
         : NDArray
             The the grid variable.
        """
        raise NotImplementedError

    def get_time_at_last_time_index(self):
        """ Wrapper for FileReader's get_time_at_last_time_index

         Returns
         -------
          : float
              The time at the last time index.
         """
        raise NotImplementedError

    def get_time_at_next_time_index(self):
        """ Wrapper for FileReader's get_time_at_next_time_index

         Returns
         -------
          : float
              The time at the last time index.
         """
        raise NotImplementedError

    def get_grid_variable_dimensions(self, var_name):
        """ Wrapper for FileReader's get_grid_variable_dimensions

        Parameters
        ----------
        var_name : str
            The name of the variable.

        Returns
        -------
         : tuple(str)
             The variable's dimensions
        """
        raise NotImplementedError

    def get_variable_dimensions(self, var_name, include_time=True):
        """ Wrapper for FileReader's get_variable_dimension

        Parameters
        ----------
        var_name : str
            The name of the variable.

        include_time : bool
            If True, the time dimension is included. Optional,
            default is True.

        Returns
        -------
         : tuple(str)
             The variable's dimensions
        """
        raise NotImplementedError

    def get_variable_shape(self, var_name, include_time=True):
        """ Wrapper for FileReader's get_variable_shape

        Parameters
        ----------
        var_name : str
            The name of the variable.
        
        include_time : bool
            If True, the time dimension is included. Optional,
            default is True.

        Returns
        -------
         : tuple(int)
             The variable's shape
        """
        raise NotImplementedError

    def get_time_dependent_variable_at_last_time_index(self, var_name, var_dims, var_type):
        """ Wrapper for FileReader's function of the same name

        Parameters
        ----------
        var_name : str
            The name of the variable.

        var_dims : tuple
             Tuple of variable dimensions.

        var_type : type
            The variable type.

        Returns
        -------
         : NDArray
             The variable array

        """
        raise NotImplementedError

    def get_time_dependent_variable_at_next_time_index(self, var_name, var_dims, var_type):
        """ Wrapper for FileReader's function of the same name

        Parameters
        ----------
        var_name : str
            The name of the variable.

        var_dims : tuple
             Tuple of variable dimensions.

        var_type : type
            The variable type.

        Returns
        -------
         : NDArray
             The variable array

        """
        raise NotImplementedError

    def get_mask_at_last_time_index(self, var_name, var_dims):
        """ Wrapper for FileReader's function of the same name

        Parameters
        ----------
        var_name : str
            The name of the variable.

        var_dims : tuple
             Tuple of variable dimensions.

        Returns
        -------
         : NDArray
             The variable array
        """
        raise NotImplementedError

    def get_mask_at_next_time_index(self, var_name, var_dims):
        """ Wrapper for FileReader's function of the same name

        Parameters
        ----------
        var_name : str
            The name of the variable.

        var_dims : tuple
             Tuple of variable dimensions.

        Returns
        -------
         : NDArray
             The variable array
        """
        raise NotImplementedError


class SerialMediator(Mediator):
    """ Serial mediator

    Serial mediator for serial runs.

    Parameters
    ----------
    config : ConfigParser
        Run configuration object

    data_source : str
        String indicating what type of data the datetime objects will be
        associated with. Options are: 'ocean', 'atmosphere', and 'wave'.

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

    def __init__(self, config, data_source, datetime_start, datetime_end):
        self.config = config

        file_name_reader = DiskFileNameReader()

        dataset_reader = NetCDFDatasetReader()

        self.file_reader = FileReader(config, data_source, file_name_reader,
                                      dataset_reader, datetime_start,
                                      datetime_end)

    def setup_data_access(self, datetime_start, datetime_end):
        self.file_reader.setup_data_access(datetime_start, datetime_end)

    def update_reading_frames(self, time):
        return self.file_reader.update_reading_frames(time)

    def get_dimension_variable(self, var_name):
        return self.file_reader.get_dimension_variable(var_name)
    
    def get_grid_variable(self, var_name, var_dims, var_type):
        return self.file_reader.get_grid_variable(var_name).astype(var_type)

    def get_time_at_last_time_index(self):
        return self.file_reader.get_time_at_last_time_index()

    def get_time_at_next_time_index(self):
        return self.file_reader.get_time_at_next_time_index()

    def get_grid_variable_dimensions(self, var_name):
        return self.file_reader.get_grid_variable_dimensions(var_name)

    def get_variable_dimensions(self, var_name, include_time=True):
        return self.file_reader.get_variable_dimensions(var_name, include_time)

    def get_variable_shape(self, var_name, include_time=True):
        return self.file_reader.get_variable_shape(var_name, include_time)

    def get_time_dependent_variable_at_last_time_index(self, var_name, var_dims, var_type):
        return self.file_reader.get_time_dependent_variable_at_last_time_index(var_name).astype(var_type)

    def get_time_dependent_variable_at_next_time_index(self, var_name, var_dims, var_type):
        return self.file_reader.get_time_dependent_variable_at_next_time_index(var_name).astype(var_type)

    def get_mask_at_last_time_index(self, var_name, var_dims):
        return self.file_reader.get_mask_at_last_time_index(var_name).astype(DTYPE_INT)

    def get_mask_at_next_time_index(self, var_name, var_dims):
        return self.file_reader.get_mask_at_next_time_index(var_name).astype(DTYPE_INT)


__all__ = ['Mediator',
           'SerialMediator']
