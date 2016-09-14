import numpy as np
from netCDF4 import Dataset, num2date
import glob
import natsort
import datetime
import logging
import ConfigParser

from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.unstruct_grid_tools import round_time, create_fvcom_grid_metrics_file

class FileReader(object):
    def __init__(self):
        pass
    
    def set_time_counters(self, start_datetime, end_datetime):
        pass

    def update_reading_frames(self):
        pass
    
    def get_dimension_variable(self):
        pass
    
    def get_grid_variable(self):
        pass
    
    def get_time_at_last_time_index(self):
        pass
    
    def get_time_at_next_time_index(self):
        pass

    def get_time_dependent_variable_at_last_time_index(self, var_name):
        pass
    
    def get_time_dependent_variable_at_next_time_index(self, var_name):
        pass

class FVCOMFileReader(FileReader):
    """
    TODO:
    
    1) Remove FVCOM specific elements?
    2) Document.
    """
    def __init__(self, config):
        self._config = config

        self._data_dir = self._config.get("OCEAN_CIRCULATION_MODEL", "data_dir")
        self._data_file_name_stem = self._config.get("OCEAN_CIRCULATION_MODEL", "data_file_stem")
        try:
            self._grid_metrics_file_name = self._config.get("OCEAN_CIRCULATION_MODEL", "grid_metrics_file")
        except ConfigParser.NoOptionError:
            self._grid_metrics_file_name = None

        # Set up grid and data access
        self._setup_file_access()

    def update_reading_frames(self, time):
        # Load the next data file, if necessary
        if (time >= self._time[-1]):
            idx = self._data_file_names.index(self._current_data_file_name) + 1
            try:
                self._current_data_file_name = self._data_file_names[idx]
            except IndexError:
                logger = logging.getLogger(__name__)
                logger.error('Failed to find the next required input data file.')
                raise
            self._open_data_file_for_reading()

        # Update time indices
        self._set_time_indices(time)

    def get_dimension_variable(self, var_name):
        return len(self._grid_file.dimensions[var_name])
        
    def get_grid_variable(self, var_name):
        return self._grid_file.variables[var_name][:]

    def get_time_at_last_time_index(self):
        return self._time[self._tidx_last]
    
    def get_time_at_next_time_index(self):
        return self._time[self._tidx_next]

    def get_time_dependent_variable_at_last_time_index(self, var_name):
        return self._current_data_file.variables[var_name][self._tidx_last,:]
    
    def get_time_dependent_variable_at_next_time_index(self, var_name):
        return self._current_data_file.variables[var_name][self._tidx_next,:]

    def _setup_file_access(self):
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
        
        # Try to read grid data from the grid metrics file, in which neighbour
        # element info (nbe) has been ordered to match node ordering in nv.
        self._grid_file = None
        if self._grid_metrics_file_name is not None:
            try:
                self._grid_file = Dataset('{}'.format(self._grid_metrics_file_name), 'r')
                logger.info('Openend grid metrics file {}.'.format(self._grid_metrics_file_name))
            except RuntimeError:
                logger.warning('Failed to read grid metrics file {}.'.format(self._grid_metrics_file_name))
        
        if not self._grid_file:
            print 'WARNING: Creating a new grid metrics file. See log file for '\
                'more information.'
            logger.info('Either no grid metrics file was given, or it could '\
                    'not be read. A new grid metrics file will now be created. '\
                    'For future simulations using the same model grid, please '\
                    'provide this file in the config -- it will save you time!')
            file_name_out = '{}/grid_metrics.nc'.format(self.config.get('GENERAL', 'out_dir'))
            create_fvcom_grid_metrics_file(self._data_file_names[0], file_name_out)
            logger.info('Created grid metrics file {}.'.format(file_name_out))
            self._grid_file = Dataset(file_name_out, 'r')
        
    def set_time_counters(self, start_datetime, end_datetime):
        """Initalise all time variables and counters.
        
        """
        
        logger = logging.getLogger(__name__)
        logger.info('Initialising all time variables and counters.')      

        # Save a reference to the new simulation start and end times for time
        # rebasing
        self._sim_datetime_s = start_datetime
        self._sim_datetime_e = end_datetime

        # Simulation start time
        rounding_interval = self._config.getint("OCEAN_CIRCULATION_MODEL", "rounding_interval")

        # Determine which data file holds data covering the simulation start time
        logger.info('Beginning search for the input data file spanning the '\
            'specified simulation start point.')  
        self._current_data_file_name = None
        for data_file_name in self._data_file_names:
            logger.info("Trying file `{}'".format(data_file_name))
            ds = Dataset(data_file_name, 'r')
            time = ds.variables['time']
            
            # Start and end time points for this file 
            data_datetime_s = round_time([num2date(time[0], units=time.units)], rounding_interval)[0]
            data_datetime_e = round_time([num2date(time[-1], units=time.units)], rounding_interval)[0]
            ds.close()
            
            if (self._sim_datetime_s >= data_datetime_s) and (self._sim_datetime_s < data_datetime_e):
                self._current_data_file_name = data_file_name
                logger.info('Found initial data file {}.'.format(self._current_data_file_name))
                break
            else:
                logger.info('Start point not found in file covering the period'\
                ' {} to {}'.format(data_datetime_s, data_datetime_e))

        # Ensure the start time is covered by the available data
        if self._current_data_file_name is None:
            raise RuntimeError('Could not find an input data file spanning the '\
                    'specified start time: {}.'.format(self._sim_datetime_s))
                
        # Check that the simulation end time is covered by the available data
        ds = Dataset(self._data_file_names[-1], 'r')
        data_datetime_e = num2date(ds.variables['time'][-1], units = ds.variables['time'].units)
        ds.close()
        
        # If the specified run time extends beyond the time period for which
        # there exists input data, raise this.
        if self._sim_datetime_e > data_datetime_e:
            raise ValueError('The specified simulation endtime {} lies '\
                    'outside of the time period for which input data is '\
                    'available. Input data is available out to '\
                    '{}.'.format(self._sim_datetime_s, self._data_datetime_e))

        # Open the current data file for reading and initialise the time array
        self._open_data_file_for_reading()

        # Set time indices for reading frames
        self._set_time_indices(0.0) # 0s as simulation start

    def _open_data_file_for_reading(self):
        """Open the current data file for reading and update time array.
        
        """
        logger = logging.getLogger(__name__)

        # Close the current data file if one has been opened previously
        if hasattr(self, '_current_data_file'):
            self._current_data_file.close()

        # Open the current data file
        try:
            self._current_data_file = Dataset(self._current_data_file_name, 'r')
            logger.info('Opened data file {} for reading.'.format(self._current_data_file_name))
        except RuntimeError:
            logger.error('Could not open data file {}.'.format(self._current_data_file_name))
            raise RuntimeError('Could not open data file for reading.')

        # Rounding interval and the simulation start time
        rounding_interval = self._config.getint("OCEAN_CIRCULATION_MODEL", "rounding_interval")

        # Read in time from the current data file and convert to a list of 
        # datetime objects. Apply rounding as specified.
        time_raw = self._current_data_file.variables['time']
        datetime_raw = num2date(time_raw[:], units=time_raw.units)
        datetime_rounded = round_time(datetime_raw, rounding_interval)
        
        # Convert to seconds using datetime_start as a reference point
        time_seconds = []
        for time in datetime_rounded:
            time_seconds.append((time - self._sim_datetime_s).total_seconds())
        self._time = np.array(time_seconds, dtype=DTYPE_FLOAT)

    def _set_time_indices(self, time):
        # Find indices for times within time_array that bracket time_start
        n_times = len(self._time)
        
        tidx_last = -1
        tidx_next = -1
        for i in xrange(0, n_times-1):
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
