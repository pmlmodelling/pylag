import numpy as np
from netCDF4 import Dataset, num2date
from datetime import timedelta
import glob
import natsort
import logging
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.data_types_python import DTYPE_FLOAT
from pylag.utils import round_time

class FileReader(object):
    """Read in and manage access to grid and field data stored in NetCDF files.

    Attributes:
    -----------
    _config : SafeConfigParser
        Run configuration object.

    _file_name_reader : FileNameReader
        Class to assist with reading in file names from disk.

    _data_dir : str
        Path to the direction containing input data

    _data_file_name_stem : str
        File name stem, used for building path names

    _grid_metrics_file_name : str
        File name or path for the grid metrics file

    _data_file_names : list[str]
        A list of input data files that were found in data_dir

    _grid_file : Dataset
        NetCDF4 dataset for the grid metrics file

    _datetime_reader : DateTimeReader
        Object to assist in reading dates/times in input data.

    _sim_datatime_s : Datetime
        The current simulation start date/time. This is not necessarily fixed
        for the lifetime of an object - it can be updated through calls
        to `setup_data_access'. This facility helps to support for the running
        of ensemble simulations.

    _sim_datatime_e : Datetime
        The current simulation end date/time. This is not necessarily fixed
        for the lifetime of an object - it can be updated through calls
        to `setup_data_access'. This facility helps to support for the running
        of ensemble simulations.

    _current_data_file_name : str
        Path to the input data file covering the current time point.

    Parameters:
    -----------
    config : SafeConfigParser
        Configuration object.

    datetime_start : Datetime
        Simulation start date/time.

    datetime_end : Datetime
        Simulation end date/time.
    """
    def __init__(self, config, file_name_reader, datetime_start, datetime_end):
        self._config = config

        self._file_name_reader = file_name_reader

        self._data_dir = self._config.get("OCEAN_CIRCULATION_MODEL", "data_dir")
        self._data_file_name_stem = self._config.get("OCEAN_CIRCULATION_MODEL", "data_file_stem")
        try:
            self._grid_metrics_file_name = self._config.get("OCEAN_CIRCULATION_MODEL", "grid_metrics_file")
        except configparser.NoOptionError:
            logger = logging.getLogger(__name__)
            logger.error('A grid metrics file was not given. Please provide '\
                'one an try again. If one needs to be generated, please '\
                'have a look at the tools provided in pylag.utils, which '\
                'provides several functions to help with the creation '\
                'of grid metrics files.')
            raise RuntimeError('A grid metrics file was not listed in the run '\
                'configuration file. See the log file for more details.')

        # Time interval between data points in input data files
        self._rounding_interval = self._config.getint("OCEAN_CIRCULATION_MODEL", "rounding_interval")

        # Initialise datetime reader
        self._datetime_reader = get_datetime_reader(config)

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
        self._data_file_names = self._file_name_reader.get_file_names(self._data_dir, self._data_file_name_stem)
    
        # Ensure files were found in the specified directory.
        if not self._data_file_names:
            raise RuntimeError('No input files found in location {}.'.format(self._data_dir))

        # Log file names
        logger.info("Found {} input data files in directory "\
            "`{}'.".format(len(self._data_file_names), self._data_dir))
        logger.info('Input data file names are: ' + ', '.join(self._data_file_names))
        
        # Open grid metrics file for reading
        logger.info('Opening grid metrics file for reading.')
        
        # Try to read grid data from the grid metrics file
        try:
            self._grid_file = Dataset('{}'.format(self._grid_metrics_file_name), 'r')
            logger.info('Openend grid metrics file {}.'.format(self._grid_metrics_file_name))
        except RuntimeError:
            logger.error('Failed to read grid metrics file {}.'.format(self._grid_metrics_file_name))
            raise ValueError('Failed to read the grid metrics file.')

        # Initialise data file names to None
        self._first_data_file_name = None
        self._second_data_file_name = None

        # Initialise data files to None
        self._first_data_file = None
        self._second_data_file = None

    def setup_data_access(self, start_datetime, end_datetime):
        """Open data files for reading and initalise all time variables.

        Use the supplied start and end times to establish which input data file(s)
        contain data spanning the specified start time. 

        Parameters:
        -----------
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
        self._sim_start_datetime = start_datetime
        self._sim_end_datetime = end_datetime

        # Determine which data file holds data covering the simulation start time
        logger.info('Beginning search for the input data file spanning the '\
            'specified simulation start point.')  

        self._first_data_file_name = None
        self._second_data_file_name = None
        for idx, data_file_name in enumerate(self._data_file_names):
            logger.info("Trying file `{}'".format(data_file_name))
            ds = Dataset(data_file_name, 'r')
        
            data_start_datetime = self._datetime_reader.get_datetime(ds, time_index=0)
            data_end_datetime = self._datetime_reader.get_datetime(ds, time_index=-1)

            ds.close()

            if data_start_datetime <= self._sim_start_datetime < data_end_datetime + timedelta(seconds=self._rounding_interval):
                self._first_data_file_name = data_file_name

                if self._sim_start_datetime < data_end_datetime:
                    self._second_data_file_name = data_file_name
                else:
                    self._second_data_file_name = self._data_file_names[idx + 1]
                    
                logger.info('Found first initial data file {}.'.format(self._first_data_file_name))
                logger.info('Found second initial data file {}.'.format(self._second_data_file_name))
                break
            else:
                logger.info('Start point not found in file covering the period'\
                ' {} to {}'.format(data_start_datetime, data_end_datetime))

        # Ensure the seach was a success
        if (self._first_data_file_name is None) or (self._second_data_file_name is None):
            raise RuntimeError('Could not find an input data file spanning the '\
                    'specified start time: {}.'.format(self._sim_start_datetime))
                
        # Open the data files for reading and initialise the time array
        self._open_data_files_for_reading()

        # Set time arrays
        self._set_time_arrays()

        # Set time indices for reading frames
        self._set_time_indices(0.0) # 0s as simulation start

    def _check_date_time_is_valid(self, date_time):
        """ Check that the given date time lies within the range covered by the input data

        Parameters:
        -----------
        date_time : Datetime
            Datetime object to check

        Returns:
        --------
         : bool
            Flag confirming whether the given date time is valid or not
        """
        ds0 = Dataset(self._data_file_names[0], 'r')
        data_datetime_0 = num2date(ds0.variables['time'][0], units = ds0.variables['time'].units)
        ds0.close()

        ds1 = Dataset(self._data_file_names[-1], 'r')
        data_datetime_1 = num2date(ds1.variables['time'][-1], units = ds1.variables['time'].units)
        ds1.close()

        if data_datetime_0 <= date_time <= data_datetime_1:
            return True

        return False

    def update_reading_frames(self, time):
        # Load data file covering the first time point, if necessary
        first_file_idx = None
        if (time < self._first_time[0]):
            first_file_idx = self._data_file_names.index(self._first_data_file_name) - 1
        elif (time >= self._first_time[-1] + self._rounding_interval):
            first_file_idx = self._data_file_names.index(self._first_data_file_name) + 1

        if first_file_idx:
            try:
                self._first_data_file_name = self._data_file_names[first_file_idx]
            except IndexError:
                logger = logging.getLogger(__name__)
                logger.error('Failed to find the next required input data file.')
                raise RuntimeError('Failed to find the next input data file.')

            self._open_first_data_file_for_reading()

            self._set_first_time_array()

        # Load data file covering the second time point, if necessary
        second_file_idx = None
        if (time < self._second_time[0] - self._rounding_interval):
            second_file_idx = self._data_file_names.index(self._second_data_file_name) - 1
        elif (time >= self._second_time[-1]):
            second_file_idx = self._data_file_names.index(self._second_data_file_name) + 1

        if second_file_idx:
            try:
                self._second_data_file_name = self._data_file_names[second_file_idx]
            except IndexError:
                logger = logging.getLogger(__name__)
                logger.error('Failed to find the next required input data file.')
                raise RuntimeError('Failed to find the next input data file.')

            self._open_second_data_file_for_reading()

            self._set_second_time_array()

        # Update time indices
        self._set_time_indices(time)

    def get_dimension_variable(self, var_name):
        return len(self._grid_file.dimensions[var_name])
        
    def get_grid_variable(self, var_name):
        return self._grid_file.variables[var_name][:].squeeze()

    def get_time_at_last_time_index(self):
        return self._first_time[self._tidx_first]

    def get_time_at_next_time_index(self):
        return self._second_time[self._tidx_second]

    def get_time_dependent_variable_at_last_time_index(self, var_name):
        return self._first_data_file.variables[var_name][self._tidx_first,:]
    
    def get_time_dependent_variable_at_next_time_index(self, var_name):
        return self._second_data_file.variables[var_name][self._tidx_second,:]

    def _open_data_files_for_reading(self):
        """Open the first and second data files for reading
        
        """
        self._open_first_data_file_for_reading()
        self._open_second_data_file_for_reading()

    def _open_first_data_file_for_reading(self):
        logger = logging.getLogger(__name__)

        # Close the first data file if one has been opened previously
        if self._first_data_file:
            self._first_data_file.close()

        # Open the first data file
        try:
            self._first_data_file = Dataset(self._first_data_file_name, 'r')
            logger.info('Opened first data file {} for reading.'.format(self._first_data_file_name))
        except RuntimeError:
            logger.error('Could not open data file {}.'.format(self._first_data_file_name))
            raise RuntimeError('Could not open data file for reading.')

    def _open_second_data_file_for_reading(self):
        logger = logging.getLogger(__name__)

        # Close the second data file if one has been opened previously
        if self._second_data_file:
            self._second_data_file.close()

        # Open the second data file
        try:
            self._second_data_file = Dataset(self._second_data_file_name, 'r')
            logger.info('Opened second data file {} for reading.'.format(self._second_data_file_name))
        except RuntimeError:
            logger.error('Could not open data file {}.'.format(self._second_data_file_name))
            raise RuntimeError('Could not open data file for reading.')

    def _set_time_arrays(self):
        self._set_first_time_array()
        self._set_second_time_array()

    def _set_first_time_array(self):
        # First time array
        # ----------------
        first_datetime = self._datetime_reader.get_datetime(self._first_data_file)

        # Convert to seconds using datetime_start as a reference point
        first_time_seconds = []
        for time in first_datetime:
            first_time_seconds.append((time - self._sim_start_datetime).total_seconds())

        self._first_time = np.array(first_time_seconds, dtype=DTYPE_FLOAT)

    def _set_second_time_array(self):
        # Second time array
        # -----------------
        second_datetime = self._datetime_reader.get_datetime(self._second_data_file)

        # Convert to seconds using datetime_start as a reference point
        second_time_seconds = []
        for time in second_datetime:
            second_time_seconds.append((time - self._sim_start_datetime).total_seconds())

        self._second_time = np.array(second_time_seconds, dtype=DTYPE_FLOAT)

    def _set_time_indices(self, time):
        # Set first time index
        # --------------------
        
        n_times = len(self._first_time)
        
        tidx_first = -1
        for i in range(0, n_times):
            t_delta = time - self._first_time[i]
            if 0.0 <= t_delta < self._rounding_interval:
                tidx_first = i
                break

        if tidx_first == -1: 
            logger = logging.getLogger(__name__)
            logger.info('The provided time {}s lies outside of the range for which '\
            'there exists input data: {} to {}s'.format(time, self._first_time[0], self._first_time[-1]))
            raise ValueError('Time out of range.')

        # Set second time index
        # ---------------------
        
        n_times = len(self._second_time)
        
        tidx_second = -1
        for i in range(0, n_times):
            t_delta = self._second_time[i] - time
            if 0.0 < t_delta <= self._rounding_interval:
                tidx_second = i
                break
            
        if tidx_second == -1: 
            logger = logging.getLogger(__name__)
            logger.info('The provided time {}s lies outside of the range for which '\
            'there exists input data: {} to {}s'.format(time, self._second_time[0], self._second_time[-1]))
            raise ValueError('Time out of range.')
        
        # Save time indices
        self._tidx_first = tidx_first
        self._tidx_second = tidx_second

# Helper classes to assist in reading file names
################################################

class FileNameReader(object):
    """ Abstract base class for FileNameReaders
    """
    def get_file_names(self, file_dir, file_name_stem):
        raise NotImplementedError

class DiskFileNameReader(object):
    """ Disk file name reader which reads in NetCDF file names from disk
    """
    def get_file_names(self, file_dir, file_name_stem):
        return natsort.natsorted(glob.glob('{}/{}*.nc'.format(file_dir, file_name_stem)))
                

# Helper classes to assist in reading dates/times
#################################################


def get_datetime_reader(config):
    """ Factory method for datetime readers

    Parameters:
    -----------
    config : SafeConfigParser
        Configuration object
    """
    data_source =  config.get("OCEAN_CIRCULATION_MODEL", "name")

    if data_source == "FVCOM":
        return FVCOMDateTimeReader(config)

    return DefaultDatetimeReader(config)


class DateTimeReader(object):
    """ Abstract base class for DateTimeReaders
    """
    def get_datetime(self, dataset, time_index=None):
        raise NotImplementedError


class DefaultDateTimeReader(DateTimeReader):

    def __init__(self, config):
        self._config = config

    def get_datetime(self, dataset, time_index=None):
        """ Get dates/times for the given dataset

        This function searches for the basic variable `time'.
        If a given source of data uses a different variable
        name or approach to saving time points, support for
        them can be added through subclassing (as with
        FVCOM) DateTimeReader.

        Parameters:
        -----------
        dataset : Dataset
            Dataset object for an FVCOM data file.
        """
        time_raw = dataset.variables['time']
        units = dataset.variables['time'].units   

        # Apply rounding
        rounding_interval = self._config.getint("OCEAN_CIRCULATION_MODEL", "rounding_interval")

        if time_index is not None:
            datetime_raw = num2date(time_raw[time_index], units=units)
            return round_time([datetime_raw], rounding_interval)[0]
        else:
            datetime_raw = num2date(time_raw[:], units=units)
            return round_time(datetime_raw, rounding_interval)


class FVCOMDateTimeReader(DateTimeReader):

    def __init__(self, config):
        self._config = config

    def get_datetime(self, dataset, time_index=None):
        """ Get FVCOM dates/times for the given dataset

        The time variable in FVCOM has the lowest precision. Instead,
        we construct the time array from the Itime and Itime2 vars,
        before then constructing datetime objects.

        Parameters:
        -----------
        dataset : Dataset
            Dataset object for an FVCOM data file.
        """
        time_raw = dataset.variables['Itime'][:] + dataset.variables['Itime2'][:] / 1000. / 60. / 60. / 24.
        units = dataset.variables['Itime'].units

        # Apply rounding
        # TODO - Confirm this is necessary when using Itime and Itime2?
        rounding_interval = self._config.getint("OCEAN_CIRCULATION_MODEL", "rounding_interval")

        if time_index is not None:
            datetime_raw = num2date(time_raw[time_index], units=units)
            return round_time([datetime_raw], rounding_interval)[0]
        else:
            datetime_raw = num2date(time_raw[:], units=units)
            return round_time(datetime_raw, rounding_interval)

