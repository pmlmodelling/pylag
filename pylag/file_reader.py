"""
A set of classes for managing access to input data, including the reading in
of data from file.
"""

import numpy as np
from netCDF4 import Dataset
from cftime import num2pydate
from datetime import timedelta
import glob
import natsort
import logging
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.data_types_python import DTYPE_FLOAT
from pylag.numerics import get_time_direction
from pylag.utils import round_time
from pylag import version


class FileReader:
    """Read in and manage access to input grid and field data

    Objects of type `FileReader` manage all access to input data stored in
    files on disk. Support for data stored in multiple files covering
    non-overlapping time intervals is included. On initialisation,
    the object will scan the specified list of input data files in order to
    find the file or files that span the specified simulation start date/time.
    Two datasets are opened - one for the each of the two input time points that
    straddle the current simulation time point. These are referred to the `first`
    and `second` data files or time points respectively, with the `first` always
    corresponding to the time point that is earlier in time than the current
    simulation time point. Time indices for the two bounding time points are also
    stored. Through calls to `update_reading_frames` both the indices corresponding
    to the bounding time points and the input datasets can be updated as the
    simulation progresses. Support for running simulations either forward or
    backward in time is included.

    Parameters
    ----------
    config : ConfigParser
        Configuration object.

    datetime_start : Datetime
        Simulation start date/time.

    datetime_end : Datetime
        Simulation end date/time.

    Attributes
    ----------
    config : ConfigParser
        Run configuration object.

    file_name_reader : FileNameReader
        Object to assist with reading in file names from disk

    dataset_reader : DatasetReader
        Object to assist with reading in NetCDF4 datasets

    datetime_reader : DateTimeReader
        Object to assist with reading dates/times in input data.

    data_dir : str
        Path to the directory containing input data

    data_file_name_stem : str
        File name stem, used for building path names

    grid_metrics_file_name : str
        File name or path to the grid metrics file

    grid_file : Dataset
        NetCDF4 grid metrics dataset

    data_file_names : list[str]
        A list of input data files that were found in `data_dir`

    first_data_file_name : str
        Name of the data file containing the `first` time point bounding the
        current point in time.

    second_data_file_name : str
        Name of data file containing the `second` time point bounding the
        current point in time.

    first_data_file : Dataset
        Dataset containing the `first` time point bounding the
        current point in time.

    second_data_file : Dataset
        Dataset containing the `second` time point bounding the
        current point in time.

    time_direction : int
        Flag indicating the direction of integration. 1 forward, -1 backward.

    first_time : array_like[float]
        Time array containing the `first` time point bounding the
        current point in time.

    second_time : array_like[float]
        Time array containing the `second` time point bounding the
        current point in time.

    tidx_first : int
        Array index corresponding to the `first` time point bounding
        the current point in time.

    tidx_second : int
        Array index corresponding to the `second` time point bounding
        the current point in time.

    sim_start_datatime : Datetime
        The current simulation start date/time. This is not necessarily fixed
        for the lifetime of the object - it can be updated through calls
        to `setup_data_access`. This helps support the running
        of ensemble simulations.

    sim_end_datatime : Datetime
        The current simulation end date/time. This is not necessarily fixed
        for the lifetime of the object - it can be updated through calls
        to `setup_data_access`. This helps support the running
        of ensemble simulations.

    """
    def __init__(self, config, file_name_reader, dataset_reader, datetime_start, datetime_end):
        self.config = config

        self.file_name_reader = file_name_reader

        self.dataset_reader = dataset_reader

        self.data_dir = self.config.get("OCEAN_CIRCULATION_MODEL", "data_dir")
        self.data_file_name_stem = self.config.get("OCEAN_CIRCULATION_MODEL", "data_file_stem")
        try:
            self.grid_metrics_file_name = self.config.get("OCEAN_CIRCULATION_MODEL", "grid_metrics_file")
        except configparser.NoOptionError:
            logger = logging.getLogger(__name__)
            logger.error('A grid metrics file was not given. Please provide '\
                'one an try again. If one needs to be generated, please '\
                'have a look at the tools provided in pylag.utils, which '\
                'provides several functions to help with the creation '\
                'of grid metrics files.')
            raise RuntimeError('A grid metrics file was not listed in the run '\
                'configuration file. See the log file for more details.')

        # Time variable name
        try:
            self._time_var_name = self.config.get("OCEAN_CIRCULATION_MODEL", "time_var_name").strip()
        except configparser.NoOptionError:
            self._time_var_name = "time"

        # Time direction
        self.time_direction = int(get_time_direction(config))

        # Initialise datetime reader
        self.datetime_reader = get_datetime_reader(config)

        # Read in grid info. and search for input data files.
        self._setup_file_access()

        # Setup data access using the given simulation start and end datetimes
        self.setup_data_access(datetime_start, datetime_end)

    def _setup_file_access(self):
        """ Set up access to input data files

        This method is called from __init__() during class initialisation.

        The following instance variables are defined here:

            _data_file_names - A list holding paths to input data files.

            _grid_file - NetCDF4 dataset for the model's grid metrics file.
        """

        logger = logging.getLogger(__name__)
        
        # First save output file names into a list
        logger.info('Searching for input data files.')
        self.data_file_names = self.file_name_reader.get_file_names(self.data_dir, self.data_file_name_stem)
    
        # Ensure files were found in the specified directory.
        if not self.data_file_names:
            raise RuntimeError('No input files found in location {}.'.format(self.data_dir))
        else:
            self.n_data_files = len(self.data_file_names)

        # Log file names
        logger.info("Found {} input data files in directory "\
            "`{}'.".format(self.n_data_files, self.data_dir))
        logger.info('Input data file names are: ' + ', '.join(self.data_file_names))
        
        # Open grid metrics file for reading
        logger.info('Opening grid metrics file for reading.')
        
        # Try to read grid data from the grid metrics file
        try:
            self.grid_file = self.dataset_reader.read_dataset(self.grid_metrics_file_name)
            logger.info('Opened grid metrics file {}.'.format(self.grid_metrics_file_name))

            if self.grid_file.getncattr('pylag-version-id') != version.git_revision:
                logger.warning('The grid metrics file was created with a different version of PyLag to that ' \
                               'being run. To avoid consistency issues, please update the grid metrics file.')
        except RuntimeError:
            logger.error('Failed to read grid metrics file {}.'.format(self.grid_metrics_file_name))
            raise ValueError('Failed to read the grid metrics file.')

        # Initialise data file names to None
        self.first_data_file_name = None
        self.second_data_file_name = None

        # Initialise data files to None
        self.first_data_file = None
        self.second_data_file = None

    def setup_data_access(self, start_datetime, end_datetime):
        """Open data files for reading and initalise all time variables

        Use the supplied start and end times to establish which input data file(s)
        contain data spanning the specified start time. 

        Parameters
        ----------
        start_datetime : Datetime
            Simulation start date/time.

        end_datetime : Datetime
            Simulation end date/time.
        """
        
        logger = logging.getLogger(__name__)
        logger.info('Setting up input data access.')

        if not self._check_date_time_is_valid(start_datetime):
            raise ValueError('The start date/time {} lies '\
                    'outside of the time period for which input data is '\
                    'available.'.format(start_datetime))

        if not self._check_date_time_is_valid(end_datetime):
            raise ValueError('The end date/time {} lies '\
                    'outside of the time period for which input data is '\
                    'available.'.format(end_datetime))
        
        # Save a reference to the simulation start time for time rebasing
        self.sim_start_datetime = start_datetime
        self.sim_end_datetime = end_datetime

        # Determine which data file holds data covering the simulation start time
        logger.info('Beginning search for the input data file spanning the '\
            'specified simulation start point.')  

        # Check for unusable input data
        ds_first = self.dataset_reader.read_dataset(self.data_file_names[0])
        datetimes_first = self.datetime_reader.get_datetime(ds_first)
        if self.n_data_files == 1 and len(datetimes_first) == 1:
            logger.info('The single input data file found contains just a single time point '\
                        'which is insufficient to perform a simulation.')
            raise RuntimeError('Only one time point value found in input dataset')

        self.first_data_file_name = None
        self.second_data_file_name = None
        for idx, data_file_name in enumerate(self.data_file_names):
            logger.info("Trying file `{}'".format(data_file_name))
            ds = self.dataset_reader.read_dataset(data_file_name)
        
            data_start_datetime = self.datetime_reader.get_datetime(ds, time_index=0)
            data_end_datetime = self.datetime_reader.get_datetime(ds, time_index=-1)

            # Compute time delta
            time_delta = self.compute_time_delta_between_datasets(data_file_name, forward=True)

            ds.close()

            if data_start_datetime <= self.sim_start_datetime < data_end_datetime + timedelta(seconds=time_delta):
                # Set file names depending on time direction
                if self.time_direction == 1:

                    self.first_data_file_name = data_file_name

                    if self.sim_start_datetime < data_end_datetime:
                        self.second_data_file_name = data_file_name
                    else:
                        self.second_data_file_name = self.data_file_names[idx + 1]
                else:

                    if self.sim_start_datetime == data_start_datetime:
                        self.first_data_file_name = self.data_file_names[idx - 1]
                        self.second_data_file_name = data_file_name
                    else:
                        if self.sim_start_datetime <= data_end_datetime:
                            self.first_data_file_name = data_file_name
                            self.second_data_file_name = data_file_name
                        else:
                            self.first_data_file_name = data_file_name
                            self.second_data_file_name = self.data_file_names[idx + 1]
                    
                logger.info('Found first initial data file {}.'.format(self.first_data_file_name))
                logger.info('Found second initial data file {}.'.format(self.second_data_file_name))
                break
            else:
                logger.info('Start point not found in file covering the period'\
                ' {} to {}'.format(data_start_datetime, data_end_datetime))

        # Ensure the search was a success
        if (self.first_data_file_name is None) or (self.second_data_file_name is None):
            raise RuntimeError('Could not find an input data file spanning the '\
                    'specified start time: {}.'.format(self.sim_start_datetime))
                
        # Open the data files for reading and initialise the time array
        self._open_data_files_for_reading()

        # Set time arrays
        self._set_time_arrays()

        # Set time indices for reading frames
        self._set_time_indices(0.0) # 0s as simulation start

    def _check_date_time_is_valid(self, date_time):
        """ Check that the given date time lies within the range covered by the input data

        Parameters
        ----------
        date_time : Datetime
            Datetime object to check

        Returns
        --------
         : bool
            Flag confirming whether the given date time is valid or not
        """
        ds0 = self.dataset_reader.read_dataset(self.data_file_names[0])
        data_datetime_0 = num2pydate(ds0.variables[self._time_var_name][0], units=ds0.variables[self._time_var_name].units)
        ds0.close()

        ds1 = self.dataset_reader.read_dataset(self.data_file_names[-1])
        data_datetime_1 = num2pydate(ds1.variables[self._time_var_name][-1], units=ds1.variables[self._time_var_name].units)
        ds1.close()

        if data_datetime_0 <= date_time < data_datetime_1:
            return True

        return False

    def compute_time_delta_between_datasets(self, data_file_name, forward):
        """ Compute time delta between datasets

        If there is only one dataset or the last data file is given a value
        of zero is returned. Otherwise, time delta is the time difference in
        seconds between the last (first) time point in the named data file and the
        first (last) time point in the next (previous) data file, as stored in
        `self.data_file_names`. The forward argument is used to determine whether
        time delta is computed as the difference between the next or last files.

        Parameters
        ----------
        data_file_name : str
            Dataset file name.

        forward : bool
            If True, compute time delta between the last time point in the current
            file and the first time point in the next file. If False, compute
            time delta between the first time point in the current file and the
            last time point in the previous file.

        Returns
        -------
        time_delta : float
            The absolute time difference in seconds.
        """
        if self.n_data_files == 1:
            # There is only one file or we are searching the last file in the set
            # so we set time_delta to zero.
            return 0.0

        # Array index of the given data file
        file_idx_a = self.data_file_names.index(data_file_name)

        # Set other indices depending on the value of `forward`
        if forward:
            if file_idx_a == self.n_data_files - 1:
                # Last file in list - return zero
                return 0.0

            file_idx_b = file_idx_a + 1
            time_index_a = -1
            time_index_b = 0
        else:
            if file_idx_a == 0:
                # First file in list - return zero
                return 0.0

            file_idx_b = file_idx_a - 1
            time_index_a = 0
            time_index_b = -1

        ds_a = self.dataset_reader.read_dataset(data_file_name)
        datetime_a = self.datetime_reader.get_datetime(ds_a, time_index=time_index_a)
        ds_a.close()

        ds_b = self.dataset_reader.read_dataset(self.data_file_names[file_idx_b])
        datetime_b = self.datetime_reader.get_datetime(ds_b, time_index=time_index_b)
        ds_b.close()

        return abs((datetime_b - datetime_a).total_seconds())

    def update_reading_frames(self, time):
        """ Update input datasets and reading frames

        Update input datasets and reading frames using the given `time`, which
        is the current simulation time in seconds.

        Parameters
        ----------
        time : float
            Time

        """
        # Compute time delta
        time_delta = self.compute_time_delta_between_datasets(self.first_data_file_name, forward=True)

        # Load data file covering the first time point, if necessary
        first_file_idx = None

        if self.time_direction == 1:
            if time < self.first_time[0]:
                first_file_idx = self.data_file_names.index(self.first_data_file_name) - 1
            elif time >= self.first_time[-1] + time_delta:
                first_file_idx = self.data_file_names.index(self.first_data_file_name) + 1
        else:
            if time <= self.first_time[0]:
                first_file_idx = self.data_file_names.index(self.first_data_file_name) - 1
            elif time > self.first_time[-1] + time_delta:
                first_file_idx = self.data_file_names.index(self.first_data_file_name) + 1

        if first_file_idx is not None:
            try:
                self.first_data_file_name = self.data_file_names[first_file_idx]
            except IndexError:
                logger = logging.getLogger(__name__)
                logger.error('Failed to find the next required input data file.')
                raise RuntimeError('Failed to find the next input data file.')

            self._open_first_data_file_for_reading()

            self._set_first_time_array()

        # Compute time delta
        time_delta = self.compute_time_delta_between_datasets(self.second_data_file_name, forward=False)

        # Load data file covering the second time point, if necessary
        second_file_idx = None

        if self.time_direction == 1:
            if time < self.second_time[0] - time_delta:
                second_file_idx = self.data_file_names.index(self.second_data_file_name) - 1
            elif time >= self.second_time[-1]:
                second_file_idx = self.data_file_names.index(self.second_data_file_name) + 1
        else:
            if time <= self.second_time[0] - time_delta:
                second_file_idx = self.data_file_names.index(self.second_data_file_name) - 1
            elif time > self.second_time[-1]:
                second_file_idx = self.data_file_names.index(self.second_data_file_name) + 1

        if second_file_idx is not None:
            try:
                self.second_data_file_name = self.data_file_names[second_file_idx]
            except IndexError:
                logger = logging.getLogger(__name__)
                logger.error('Failed to find the next required input data file.')
                raise RuntimeError('Failed to find the next input data file.')

            self._open_second_data_file_for_reading()

            self._set_second_time_array()

        # Update time indices
        self._set_time_indices(time)

    def get_dimension_variable(self, var_name):
        """ Get the size of the NetCDF4 dimension variable

        Parameters
        ----------
        var_name : str
            The name of the dimension variable.

        Returns
        -------
         : int
             The size of the dimensions variable.
        """
        return len(self.grid_file.dimensions[var_name])
        
    def get_grid_variable(self, var_name):
        """ Get the NetCDF4 grid variable

        Parameters
        ----------
        var_name : str
            The name of the variable.

        Returns
        -------
         : NDArray
             The the grid variable.
        """
        return self.grid_file.variables[var_name][:].squeeze()

    def get_time_at_last_time_index(self):
        """ Get the time and the last time index

        Returns
        -------
         : float
             The time at the last time index.
        """
        return self.first_time[self.tidx_first]

    def get_time_at_next_time_index(self):
        """ Get the time and the next time index

        Returns
        -------
         : float
             The time at the next time index.
        """
        return self.second_time[self.tidx_second]

    def get_grid_variable_dimensions(self, var_name):
        """ Get the variable dimensions

        Parameters
        ----------
        var_name : str
            The name of the variable.

        Returns
        -------
         : tuple(str)
             The variable's dimensions
        """
        return self.grid_file.variables[var_name].dimensions

    def get_variable_dimensions(self, var_name):
        """ Get the variable dimensions

        Parameters
        ----------
        var_name : str
            The name of the variable.

        Returns
        -------
         : tuple(str)
             The variable's dimensions
        """
        return self.first_data_file.variables[var_name].dimensions

    def get_variable_shape(self, var_name):
        """ Get the variable shape

        Parameters
        ----------
        var_name : str
            The name of the variable.

        Returns
        -------
         : tuple(int)
             The variable's shape
        """
        return self.first_data_file.variables[var_name].shape

    def get_time_dependent_variable_at_last_time_index(self, var_name):
        """ Get the variable at the last time index

        Parameters
        ----------
        var_name : str
            The name of the variable.

        Returns
        -------
         : NDArray
             The variable array

        """
        var = self.first_data_file.variables[var_name][self.tidx_first, :]

        if np.ma.isMaskedArray(var):
            return var.filled(0.0)

        return var

    def get_time_dependent_variable_at_next_time_index(self, var_name):
        """ Get the variable at the next time index

        Parameters
        ----------
        var_name : str
            The name of the variable.

        Returns
        -------
         : NDArray
             The variable array

        """
        var = self.second_data_file.variables[var_name][self.tidx_second, :]

        if np.ma.isMaskedArray(var):
            return var.filled(0.0)

        return var

    def get_mask_at_last_time_index(self, var_name):
        """ Get the mask at the last time index

        Parameters
        ----------
        var_name : str
            The name of the variable.

        Returns
        -------
         : NDArray
             The variable mask

        """
        var = self.first_data_file.variables[var_name][self.tidx_first, :]

        if np.ma.isMaskedArray(var):
            return var.mask

        raise RuntimeError('Variable {} is not a masked array.'.format(var_name))

    def get_mask_at_next_time_index(self, var_name):
        """ Get the mask at the next time index

        Parameters
        ----------
        var_name : str
            The name of the variable.

        Returns
        -------
         : NDArray
             The variable mask

        """
        var = self.second_data_file.variables[var_name][self.tidx_second, :]

        if np.ma.isMaskedArray(var):
            return var.mask

        raise RuntimeError('Variable {} is not a masked array.'.format(var_name))

    def _open_data_files_for_reading(self):
        """Open the first and second data files for reading
        
        """
        self._open_first_data_file_for_reading()
        self._open_second_data_file_for_reading()

    def _open_first_data_file_for_reading(self):
        logger = logging.getLogger(__name__)

        # Close the first data file if one has been opened previously
        if self.first_data_file:
            self.first_data_file.close()

        # Open the first data file
        try:
            self.first_data_file = self.dataset_reader.read_dataset(self.first_data_file_name)
            logger.info('Opened first data file {} for reading.'.format(self.first_data_file_name))
        except RuntimeError:
            logger.error('Could not open data file {}.'.format(self.first_data_file_name))
            raise RuntimeError('Could not open data file for reading.')

    def _open_second_data_file_for_reading(self):
        logger = logging.getLogger(__name__)

        # Close the second data file if one has been opened previously
        if self.second_data_file:
            self.second_data_file.close()

        # Open the second data file
        try:
            self.second_data_file = self.dataset_reader.read_dataset(self.second_data_file_name)
            logger.info('Opened second data file {} for reading.'.format(self.second_data_file_name))
        except RuntimeError:
            logger.error('Could not open data file {}.'.format(self.second_data_file_name))
            raise RuntimeError('Could not open data file for reading.')

    def _set_time_arrays(self):
        self._set_first_time_array()
        self._set_second_time_array()

    def _set_first_time_array(self):
        # First time array
        # ---------------
        first_datetime = self.datetime_reader.get_datetime(self.first_data_file)

        # Convert to seconds using datetime_start as a reference point
        first_time_seconds = []
        for time in first_datetime:
            first_time_seconds.append((time - self.sim_start_datetime).total_seconds())

        self.first_time = np.array(first_time_seconds, dtype=DTYPE_FLOAT)

    def _set_second_time_array(self):
        # Second time array
        # ----------------
        second_datetime = self.datetime_reader.get_datetime(self.second_data_file)

        # Convert to seconds using datetime_start as a reference point
        second_time_seconds = []
        for time in second_datetime:
            second_time_seconds.append((time - self.sim_start_datetime).total_seconds())

        self.second_time = np.array(second_time_seconds, dtype=DTYPE_FLOAT)

    def _compute_first_dataset_time_delta(self, idx):
        # Time delta between two time points in the first time array
        # ----------------------------------------------------------
        if idx < len(self.first_time) - 1:
            return self.first_time[idx+1] - self.first_time[idx]
        else:
            return self.compute_time_delta_between_datasets(self.first_data_file_name, forward=True)

    def _compute_second_dataset_time_delta(self, idx):
        # Time delta between two time points in the second time array
        # -----------------------------------------------------------
        if idx > 0:
            return self.second_time[idx] - self.second_time[idx-1]
        else:
            return self.compute_time_delta_between_datasets(self.second_data_file_name, forward=False)

    def _set_time_indices(self, time):
        # Set first time index
        # -------------------
        
        n_times = len(self.first_time)
        
        tidx_first = -1

        if self.time_direction == 1:
            for i in range(0, n_times):
                t_delta = time - self.first_time[i]
                t_delta_dataset = self._compute_first_dataset_time_delta(i)
                if 0.0 <= t_delta < t_delta_dataset:
                    tidx_first = i
                    break
        else:
            for i in range(0, n_times):
                t_delta = time - self.first_time[i]
                t_delta_dataset = self._compute_first_dataset_time_delta(i)
                if 0.0 < t_delta <= t_delta_dataset:
                    tidx_first = i
                    break

        if tidx_first == -1: 
            logger = logging.getLogger(__name__)
            logger.info('The provided time {}s lies outside of the range for which '\
            'there exists input data: {} to {}s'.format(time, self.first_time[0], self.first_time[-1]))
            raise ValueError('Time out of range.')

        # Set second time index
        # ---------------------
        
        n_times = len(self.second_time)
        
        tidx_second = -1

        if self.time_direction == 1:
            for i in range(0, n_times):
                t_delta = self.second_time[i] - time
                t_delta_dataset = self._compute_second_dataset_time_delta(i)
                if 0.0 < t_delta <= t_delta_dataset:
                    tidx_second = i
                    break
        else:
            for i in range(0, n_times):
                t_delta = self.second_time[i] - time
                t_delta_dataset = self._compute_second_dataset_time_delta(i)
                if 0.0 <= t_delta < t_delta_dataset:
                    tidx_second = i
                    break
                
        if tidx_second == -1: 
            logger = logging.getLogger(__name__)
            logger.info('The provided time {}s lies outside of the range for which '\
            'there exists input data: {} to {}s'.format(time, self.second_time[0], self.second_time[-1]))
            raise ValueError('Time out of range.')
        
        # Save time indices
        self.tidx_first = tidx_first
        self.tidx_second = tidx_second


# Helper classes to assist in reading file names
################################################

class FileNameReader:
    """ Abstract base class for FileNameReaders

    File name readers are responsible for reading in and sorting file
    names, which will usually be stored on disk. An abstract base class
    was added in order to assist with testing FileReader's behaviour
    under circumstances when all dependencies on reading data from disk
    have been removed.
    """
    def get_file_names(self, file_dir, file_name_stem):
        """ Get file names

        Return a list of file names

        Parameters
        ----------
        file_dir : str
            Path to the input files.

        file_name_stem : str
            Unique string identifying valid input files.

        Returns
        -------
         : list[str]
             A list of file names.

        """
        raise NotImplementedError

class DiskFileNameReader(FileNameReader):
    """ Disk file name reader which reads in NetCDF file names from disk

    Derived class for reading in file names from disk.
    """
    def get_file_names(self, file_dir, file_name_stem):
        """ Get file names

        Read file names from disk. A natural sorting algorithm is applied.

        Parameters
        ----------
        file_dir : str
            Path to the input files.

        file_name_stem : str
            Unique string identifying valid input files.

        Returns
        -------
         : list[str]
             A list of file names.

        """
        return natsort.natsorted(glob.glob('{}/{}*.nc'.format(file_dir, file_name_stem)))
                

# Helper classes to assist in reading datasets
##############################################

class DatasetReader:
    """ Abstract base class for DatasetReaders

    DatasetReaders are responsible for opening and reading single Datasets.
    Abstract base class introduced to assist with testing objects of type
    FileReader.
    """
    def read_dataset(self, file_name, set_auto_mask_and_scale=True):
        """ Open a dataset for reading

        Parameters
        ----------
        file_name : str
            The name or path of the file to open

        set_auto_mask_and_scale : bool
            Flag for masking

        Returns
        -------
         : N/A
            A dataset.

        """
        raise NotImplementedError


class NetCDFDatasetReader(DatasetReader):
    """ NetCDF dataset reader

    Return a NetCDF4 dataset object.
    """
    def read_dataset(self, file_name, set_auto_maskandscale=True):
        """ Open a dataset for reading

        Parameters
        ----------
        file_name : str
            The name or path of the file to open

        set_auto_mask_and_scale : bool
            Flag for masking

        Returns
        -------
         : NetCDF4 Dataset
            A NetCDF4 dataset.

        """
        ds = Dataset(file_name, 'r')
        ds.set_auto_maskandscale(set_auto_maskandscale)
        return ds


# Helper classes to assist in reading dates/times
#################################################


def get_datetime_reader(config):
    """ Factory method for datetime readers

    Parameters
    ----------
    config : ConfigParser
        Configuration object

    Returns
    -------
     : DatetimeReader
         A DatetimeReader.
    """
    data_source = config.get("OCEAN_CIRCULATION_MODEL", "name")

    if data_source == "FVCOM":
        return FVCOMDateTimeReader(config)

    return DefaultDateTimeReader(config)


class DateTimeReader:
    """ Abstract base class for DateTimeReaders

    DatetimeReaders are responsible for reading in and processing
    datetime data within NetCDF4 datasets. Abstract base class introduced
    as datetime information is encoded in different ways in different
    datasets.
    """
    def get_datetime(self, dataset, time_index=None):
        """ Get dates/times for the given dataset

        Must be implemented in a derived class.

        Parameters
        ----------
        dataset : Dataset
            Dataset object for an FVCOM data file.

        time_index : int, optional
            The time index at which to extract data.
        """
        raise NotImplementedError


class DefaultDateTimeReader(DateTimeReader):
    """ Default Datetime reader

    Default datetime readers read in datetime information from a single variable
    in the supplied NetCDF dataset. The name of the time variable should be given
    in the run config file. If one is not given, it defaults to the name `time`.

    Parameters
    ----------
    config : ConfigParser
        A run configuration object,
    """

    def __init__(self, config):
        self.config = config

        # Time variable name
        try:
            self._time_var_name = self.config.get("OCEAN_CIRCULATION_MODEL", "time_var_name").strip()
        except configparser.NoOptionError:
            self._time_var_name = "time"

    def get_datetime(self, dataset, time_index=None):
        """ Get dates/times for the given dataset

        This function searches for the basic variable `time`.
        If a given source of data uses a different variable
        name or approach to saving time points, support for
        them can be added through subclassing (as with
        FVCOM) DateTimeReader.

        Parameters
        ----------
        dataset : Dataset
            Dataset object for an FVCOM data file.

        time_index : int, optional
            The time index at which to extract data. Default behaviour is to return
            the full time array as datetime objects.

        Returns
        -------
         : list[datetime]
             If `time_index` is None, return a full list of datetime objects.

         : Datetime
             If `time_index` is not None, a single datetime object.
        """
        time_raw = dataset.variables[self._time_var_name]
        units = dataset.variables[self._time_var_name].units

        # Apply rounding
        rounding_interval = self.config.getint("OCEAN_CIRCULATION_MODEL", "rounding_interval")

        if time_index is not None:
            datetime_raw = num2pydate(time_raw[time_index], units=units)
            return round_time([datetime_raw], rounding_interval)[0]
        else:
            datetime_raw = num2pydate(time_raw[:], units=units)
            return round_time(datetime_raw, rounding_interval)


class FVCOMDateTimeReader(DateTimeReader):
    """ FVCOM Datetime reader

    FVCOM datetime readers read in datetime information from a NetCDF input
    file generated by FVCOM.

    Parameters
    ----------
    config : ConfigParser
        A run configuration object,
    """

    def __init__(self, config):
        self.config = config

    def get_datetime(self, dataset, time_index=None):
        """ Get FVCOM dates/times for the given dataset

        The time variable in FVCOM has the lowest precision. Instead,
        we construct the time array from the Itime and Itime2 vars,
        before then constructing datetime objects.

        Parameters
        ----------
        dataset : Dataset
            Dataset object for an FVCOM data file.

        Returns
        -------
         : list[datetime]
             If `time_index` is None, return a full list of datetime objects.

         : Datetime
        """
        time_raw = dataset.variables['Itime'][:] + dataset.variables['Itime2'][:] / 1000. / 60. / 60. / 24.
        units = dataset.variables['Itime'].units

        # Apply rounding
        # TODO - Confirm this is necessary when using Itime and Itime2?
        rounding_interval = self.config.getint("OCEAN_CIRCULATION_MODEL", "rounding_interval")

        if time_index is not None:
            datetime_raw = num2pydate(time_raw[time_index], units=units)
            return round_time([datetime_raw], rounding_interval)[0]
        else:
            datetime_raw = num2pydate(time_raw[:], units=units)
            return round_time(datetime_raw, rounding_interval)


__all__ = ["FileReader",
           "FileNameReader",
           "DiskFileNameReader",
           "DatasetReader",
           "NetCDFDatasetReader",
           "DateTimeReader",
           "DefaultDateTimeReader",
           "FVCOMDateTimeReader",
           "get_datetime_reader"]
