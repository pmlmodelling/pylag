import numpy as np
from netCDF4 import Dataset, num2date
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
    def __init__(self, config, datetime_start, datetime_end):
        self._config = config

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
        self._data_file_names = natsort.natsorted(glob.glob('{}/{}*.nc'.format(self._data_dir, 
                self._data_file_name_stem)))
                
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

        # Initialise data file to None
        self._current_data_file = None

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
        self._sim_datetime_s = start_datetime
        self._sim_datetime_e = end_datetime

        # Determine which data file holds data covering the simulation start time
        logger.info('Beginning search for the input data file spanning the '\
            'specified simulation start point.')  

        self._current_data_file_name = None
        for data_file_name in self._data_file_names:
            logger.info("Trying file `{}'".format(data_file_name))
            ds = Dataset(data_file_name, 'r')
        
            data_start_datetime = self._datetime_reader.get_datetime(ds, time_index=0)
            data_end_datetime = self._datetime_reader.get_datetime(ds, time_index=-1)

            ds.close()

            if (self._sim_datetime_s >= data_start_datetime) and (self._sim_datetime_s < data_end_datetime):
                self._current_data_file_name = data_file_name
                logger.info('Found initial data file {}.'.format(self._current_data_file_name))
                break
            else:
                logger.info('Start point not found in file covering the period'\
                ' {} to {}'.format(data_start_datetime, data_end_datetime))

        # Ensure the seach was a success
        if self._current_data_file_name is None:
            raise RuntimeError('Could not find an input data file spanning the '\
                    'specified start time: {}.'.format(self._sim_datetime_s))
                
        # Open the current data file for reading and initialise the time array
        self._open_data_file_for_reading()

        # Set time arrays
        self._set_time_array()

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
        # Load the next data file, if necessary
        if (time >= self._time[-1]):
            idx = self._data_file_names.index(self._current_data_file_name) + 1
            try:
                self._current_data_file_name = self._data_file_names[idx]
            except IndexError:
                logger = logging.getLogger(__name__)
                logger.error('Failed to find the next required input data file.')
                raise RuntimeError('Failed to find the next input data file.')

            self._open_data_file_for_reading()

            self._set_time_array()

        # Update time indices
        self._set_time_indices(time)

    def get_dimension_variable(self, var_name):
        return len(self._grid_file.dimensions[var_name])
        
    def get_grid_variable(self, var_name):
        return self._grid_file.variables[var_name][:].squeeze()

    def get_time_at_last_time_index(self):
        return self._time[self._tidx_last]

    def get_time_at_next_time_index(self):
        return self._time[self._tidx_next]

    def get_time_dependent_variable_at_last_time_index(self, var_name):
        return self._current_data_file.variables[var_name][self._tidx_last,:]
    
    def get_time_dependent_variable_at_next_time_index(self, var_name):
        return self._current_data_file.variables[var_name][self._tidx_next,:]

    def _open_data_file_for_reading(self):
        """Open the current data file for reading and update time array.
        
        """
        logger = logging.getLogger(__name__)

        # Close the current data file if one has been opened previously
        if self._current_data_file:
            self._current_data_file.close()

        # Open the current data file
        try:
            self._current_data_file = Dataset(self._current_data_file_name, 'r')
            logger.info('Opened data file {} for reading.'.format(self._current_data_file_name))
        except RuntimeError:
            logger.error('Could not open data file {}.'.format(self._current_data_file_name))
            raise RuntimeError('Could not open data file for reading.')

    def _set_time_array(self):
        datetime = self._datetime_reader.get_datetime(self._current_data_file)

        # Convert to seconds using datetime_start as a reference point
        time_seconds = []
        for time in datetime:
            time_seconds.append((time - self._sim_datetime_s).total_seconds())

        self._time = np.array(time_seconds, dtype=DTYPE_FLOAT)

    def _set_time_indices(self, time):
        # Find indices for times within time_array that bracket time_start
        n_times = len(self._time)
        
        tidx_last = -1
        tidx_next = -1
        for i in range(0, n_times-1):
            if time >= self._time[i] and time < self._time[i+1]:
                tidx_last = i
                tidx_next = tidx_last + 1
                break

        if tidx_last == -1:
            logger = logging.getLogger(__name__)
            logger.info('The provided time {}s lies outside of the range for which '\
            'there exists input data: {} to {}s'.format(time, self._time[0], self._time[-1]))
            raise ValueError('Time out of range.')
        
        # Save time indices
        self._tidx_last = tidx_last
        self._tidx_next = tidx_next

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
    """ Abstract base class for DatetimeReaders
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

