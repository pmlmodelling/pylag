"""
A set of classes for managing access to input data, including the reading in
of data from file.
"""

import numpy as np
from netCDF4 import Dataset
from datetime import timedelta
import glob
import natsort
import logging
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.exceptions import PyLagValueError, PyLagRuntimeError
from pylag.data_types_python import DTYPE_FLOAT
from pylag.numerics import get_global_time_step, get_time_direction
from pylag.datetime_reader import get_datetime_reader
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

    data_source : str
        String indicating what type of data the datetime objects will be
        associated with. Options are: 'ocean', 'atmosphere', and 'wave'.

    file_name_reader : FileNameReader
        Object to assist with reading in file names.

    dataset_reader : DatasetReader
        Object to assist with reading in datasets

    datetime_start : Datetime
        Simulation start date/time.

    datetime_end : Datetime
        Simulation end date/time.

    Attributes
    ----------
    config : ConfigParser
        Run configuration object.

    config_section_name : str
        String identifying the section of the config where parameters
        describing the data are listed (e.g. WAVE_DATA).

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
    def __init__(self, config, data_source, file_name_reader, dataset_reader,
                 datetime_start, datetime_end):
        self.config = config

        # Determine the appropriate section config name from the data source
        if data_source == 'ocean':
            self.config_section_name = 'OCEAN_DATA'
        elif data_source == 'atmosphere':
            self.config_section_name = 'ATMOSPHERE_DATA'
        elif data_source == 'wave':
            self.config_section_name = 'WAVE_DATA'
        else:
            raise PyLagValueError(f"Unsupported data source `{data_source}. "
                                  f"Valid options are `ocean`, `atmosphere` "
                                  f"and `wave`.")

        self.file_name_reader = file_name_reader

        self.dataset_reader = dataset_reader

        self.data_dir = self.config.get(self.config_section_name,
                                        "data_dir")
        self.data_file_name_stem = self.config.get(self.config_section_name,
                                                   "data_file_stem")
        try:
            self.grid_metrics_file_name = self.config.get(
                self.config_section_name, "grid_metrics_file")
        except configparser.NoOptionError:
            logger = logging.getLogger(__name__)
            logger.error(f"A grid metrics file was not given. Please provide "
                         f"one and try again. If one needs to be generated, "
                         f"please take a look at PyLag's online documentation.")
            raise PyLagRuntimeError(f"A grid metrics file was not listed in "
                                    f"the run configuration file. See the log "
                                    f"file for more details.")

        # Time dimension name
        try:
            self._time_dim_name = self.config.get(self.config_section_name,
                                                  "time_dim_name").strip()
        except configparser.NoOptionError:
            # Adopt default name `time`
            self._time_dim_name = "time"

        # Time direction
        self.time_direction = int(get_time_direction(config))

        # Initialise datetime reader
        self.datetime_reader = get_datetime_reader(config,
                                                   self.config_section_name)

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
        self.data_file_names = self.file_name_reader.get_file_names(
                self.data_dir, self.data_file_name_stem)
    
        # Ensure files were found in the specified directory.
        if not self.data_file_names:
            raise PyLagRuntimeError(f"No input files found in "
                                    f"location {self.data_dir}.")
        else:
            self.n_data_files = len(self.data_file_names)

        # Log file names
        logger.info(f"Found {self.n_data_files} input data files in directory "
                    f"`{self.data_dir}'.")
        logger.info(f"Input data file names "
                    f"are:" + ", ".join(self.data_file_names))
        
        # Open grid metrics file for reading
        logger.info("Opening grid metrics file for reading.")
        
        # Try to read grid data from the grid metrics file
        try:
            self.grid_file = self.dataset_reader.read_dataset(
                    self.grid_metrics_file_name)
            logger.info(f"Opened grid metrics file "
                        f"{self.grid_metrics_file_name}.")

            try:
                if self.grid_file.getncattr('pylag-version-id') != \
                        version.git_revision:
                    logger.warning(f"The grid metrics file was created with a "
                                   f"different version of PyLag to that being "
                                   f"run. To avoid consistency issues, please "
                                   f"update the grid metrics file.")
            except AttributeError:
                pass
        except RuntimeError:
            logger.error(f"Failed to read grid metrics file "
                         f"`{self.pylag_grid_metrics_file_name}`.")
            raise PyLagValueError("Failed to read the grid metrics file.")

        # Initialise data file names to None
        self.first_data_file_name = None
        self.second_data_file_name = None

        # Initialise data files to None
        self.first_data_file = None
        self.second_data_file = None

    def setup_data_access(self, start_datetime, end_datetime):
        """Open data files for reading and initalise all time variables

        Use the supplied start and end times to establish which input data
        file(s) contain data spanning the specified start time.

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
            raise PyLagValueError(f"The start date/time {start_datetime} lies "
                                  f"outside of the time period for which input "
                                  f"data is available.")

        if not self._check_date_time_is_valid(end_datetime):
            raise PyLagValueError(f"The end date/time {end_datetime} lies "
                                  f"outside of the time period for which input "
                                  f"data is available.")
        
        # Save a reference to the simulation start time for time rebasing
        self.sim_start_datetime = start_datetime
        self.sim_end_datetime = end_datetime

        # Determine which data file holds data covering the simulation start
        logger.info(f"Beginning search for the input data file spanning the "
                    f"specified simulation start point.")

        # Check for unusable input data
        ds_first = self.dataset_reader.read_dataset(self.data_file_names[0])
        datetimes_first = self.datetime_reader.get_datetime(ds_first)
        if self.n_data_files == 1 and len(datetimes_first) == 1:
            logger.info(f"The single input data file found contains just a "
                        f"single time point which is insufficient to perform "
                        f"a simulation.")
            raise PyLagRuntimeError(f"Only one time point value found in "
                                    f"input dataset")

        self.first_data_file_name = None
        self.second_data_file_name = None
        for idx, data_file_name in enumerate(self.data_file_names):
            logger.info(f"Trying file `{data_file_name}'")
            ds = self.dataset_reader.read_dataset(data_file_name)
        
            data_start_datetime = self.datetime_reader.get_datetime(ds,
                    time_index=0)
            data_end_datetime = self.datetime_reader.get_datetime(ds,
                    time_index=-1)

            # Compute time delta
            time_delta = self.compute_time_delta_between_datasets(
                    data_file_name, forward=True)

            ds.close()

            if (data_start_datetime <= self.sim_start_datetime <
                    data_end_datetime + timedelta(seconds=time_delta)):
                # Set file names depending on time direction
                if self.time_direction == 1:

                    self.first_data_file_name = data_file_name

                    if self.sim_start_datetime < data_end_datetime:
                        self.second_data_file_name = data_file_name
                    else:
                        self.second_data_file_name = \
                                self.data_file_names[idx + 1]
                else:

                    if self.sim_start_datetime == data_start_datetime:
                        self.first_data_file_name = \
                                self.data_file_names[idx - 1]
                        self.second_data_file_name = data_file_name
                    else:
                        if self.sim_start_datetime <= data_end_datetime:
                            self.first_data_file_name = data_file_name
                            self.second_data_file_name = data_file_name
                        else:
                            self.first_data_file_name = data_file_name
                            self.second_data_file_name = \
                                    self.data_file_names[idx + 1]
                    
                logger.info(f"Found first initial data file "
                            f"{self.first_data_file_name}.")
                logger.info(f"Found second initial data file "
                            f"{self.second_data_file_name}.")
                break
            else:
                logger.info(f"Start point not found in file covering the "
                            f"period {data_start_datetime} to "
                            f"{data_end_datetime}")

        # Ensure the search was a success
        if (self.first_data_file_name is None) or \
                (self.second_data_file_name is None):
            raise PyLagRuntimeError(f'Could not find an input data file '
                                    f'spanning the specified start time: '
                                    f'{self.sim_start_datetime}.')
                
        # Open the data files for reading and initialise the time array
        self._open_data_files_for_reading()

        # Set time arrays
        self._set_time_arrays()

        # Set time indices for reading frames
        self._set_time_indices(0.0) # 0s as simulation start

        # Check the choice of start time and time step yields an even number
        # of time steps between the start time and the times at which data are
        # defined. We check against both the first and second times when input
        # data are defined to ensure the check is robust.
        time_step = get_global_time_step(self.config)
        n_steps_before = self.first_time[self.tidx_first] / time_step
        n_steps_after = self.second_time[self.tidx_second] / time_step
        if not (n_steps_before.is_integer() and n_steps_after.is_integer()):
            raise PyLagValueError(f'PyLag requires there to be an integer '
                                  f'number of time steps (measured in seconds) '
                                  f'between the simulation start time and the '
                                  f'times when input data are defined. '
                                  f'Please modify your start time or time '
                                  f'step to ensure this is the case.')

    def _check_date_time_is_valid(self, date_time):
        """ Check the given date lies within the range covered by the input data

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
        data_datetime_0 = self.datetime_reader.get_datetime(ds0, time_index=0)
        ds0.close()

        ds1 = self.dataset_reader.read_dataset(self.data_file_names[-1])
        data_datetime_1 = self.datetime_reader.get_datetime(ds1, time_index=-1)
        ds1.close()

        if data_datetime_0 <= date_time < data_datetime_1:
            return True

        return False

    def compute_time_delta_between_datasets(self, data_file_name, forward):
        """ Compute time delta between datasets

        If there is only one dataset or the last data file is given a value
        of zero is returned. Otherwise, time delta is the time difference in
        seconds between the last (first) time point in the data file and the
        first (last) time point in the next (previous) data file, as stored in
        `self.data_file_names`. The forward argument is used to determine if
        time delta is computed as the difference between the next or last files.

        Parameters
        ----------
        data_file_name : str
            Dataset file name.

        forward : bool
            If True, compute time delta between the last time point in the
            current file and the first time point in the next file. If False,
            compute time delta between the first time point in the current file
            and the last time point in the previous file.

        Returns
        -------
        time_delta : float
            The absolute time difference in seconds.
        """
        if self.n_data_files == 1:
            # There is only one file or we are searching the last file in the
            # set so we set time_delta to zero.
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
        datetime_a = self.datetime_reader.get_datetime(ds_a,
                                                       time_index=time_index_a)
        ds_a.close()

        ds_b = self.dataset_reader.read_dataset(
                self.data_file_names[file_idx_b])
        datetime_b = self.datetime_reader.get_datetime(ds_b,
                                                       time_index=time_index_b)
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
        time_delta = self.compute_time_delta_between_datasets(
                self.first_data_file_name, forward=True)

        # Load data file covering the first time point, if necessary
        first_file_idx = None

        if self.time_direction == 1:
            if time < self.first_time[0]:
                first_file_idx = self.data_file_names.index(
                    self.first_data_file_name) - 1
            elif time >= self.first_time[-1] + time_delta:
                first_file_idx = self.data_file_names.index(
                    self.first_data_file_name) + 1
        else:
            if time <= self.first_time[0]:
                first_file_idx = self.data_file_names.index(
                    self.first_data_file_name) - 1
            elif time > self.first_time[-1] + time_delta:
                first_file_idx = self.data_file_names.index(
                    self.first_data_file_name) + 1

        if first_file_idx is not None:
            try:
                self.first_data_file_name = self.data_file_names[first_file_idx]
            except IndexError:
                logger = logging.getLogger(__name__)
                logger.error(f'Failed to find the next input data file.')
                raise PyLagRuntimeError(f'Failed to find the next input '
                                        f'data file.')

            self._open_first_data_file_for_reading()

            self._set_first_time_array()

        # Compute time delta
        time_delta = self.compute_time_delta_between_datasets(
                self.second_data_file_name, forward=False)

        # Load data file covering the second time point, if necessary
        second_file_idx = None

        if self.time_direction == 1:
            if time < self.second_time[0] - time_delta:
                second_file_idx = self.data_file_names.index(
                    self.second_data_file_name) - 1
            elif time >= self.second_time[-1]:
                second_file_idx = self.data_file_names.index(
                    self.second_data_file_name) + 1
        else:
            if time <= self.second_time[0] - time_delta:
                second_file_idx = self.data_file_names.index(
                    self.second_data_file_name) - 1
            elif time > self.second_time[-1]:
                second_file_idx = self.data_file_names.index(
                    self.second_data_file_name) + 1

        if second_file_idx is not None:
            try:
                self.second_data_file_name = \
                    self.data_file_names[second_file_idx]
            except IndexError:
                logger = logging.getLogger(__name__)
                logger.error(f'Failed to find the next input data file.')
                raise PyLagRuntimeError(f'Failed to find the next input '
                                        f'data file.')

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
        return np.ascontiguousarray(
            self.grid_file.variables[var_name][:].squeeze())

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

    def get_variable_dimensions(self, var_name, include_time=True):
        """ Get the variable dimensions

        Parameters
        ----------
        var_name : str
            The name of the variable.

        include_time : bool
            If False, the time dimension is not included in the dimensions.
            Optional, default: True.

        Returns
        -------
         : tuple(str)
             The variable's dimensions
        """
        if include_time:
            return self.first_data_file.variables[var_name].dimensions
        else:
            dimensions = self.first_data_file.variables[var_name].dimensions
            dimensions = list(dimensions)
            dimensions.remove(self._time_dim_name)
            return tuple(dimensions)

    def get_variable_shape(self, var_name, include_time=True):
        """ Get the variable shape

        Parameters
        ----------
        var_name : str
            The name of the variable.

        include_time : bool
            If False, the time dimension is not included in the shape.
            Optional, default: True.

        Returns
        -------
         : tuple(int)
             The variable's shape
        """
        if include_time:
            return self.first_data_file.variables[var_name].shape
        else:
            dimensions = self.get_variable_dimensions(var_name)
            time_dim_idx = dimensions.index(self._time_dim_name)

            shape = list(self.first_data_file.variables[var_name].shape)
            shape.pop(time_dim_idx)

            return tuple(shape)

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
        # Get time dimension index
        var_dims = self.get_variable_dimensions(var_name)
        time_dim_idx = var_dims.index(self._time_dim_name)

        # Get variable
        nc_var = self.first_data_file.variables[var_name]
        var = self._get_time_slice(nc_var, time_dim_idx, self.tidx_first)

        if np.ma.isMaskedArray(var):
            var = var.filled(0.0)

        return np.ascontiguousarray(var)

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
        # Get time dimension index
        var_dims = self.get_variable_dimensions(var_name)
        time_dim_idx = var_dims.index(self._time_dim_name)

        # Get variable
        nc_var = self.second_data_file.variables[var_name]
        var = self._get_time_slice(nc_var, time_dim_idx, self.tidx_second)

        if np.ma.isMaskedArray(var):
            var = var.filled(0.0)

        return np.ascontiguousarray(var)

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
        # Get time dimension index
        var_dims = self.get_variable_dimensions(var_name)
        time_dim_idx = var_dims.index(self._time_dim_name)

        # Get variable
        nc_var = self.first_data_file.variables[var_name]
        var = self._get_time_slice(nc_var, time_dim_idx, self.tidx_first)

        if np.ma.isMaskedArray(var):
            return np.ascontiguousarray(var.mask)

        raise PyLagRuntimeError(f'Variable {var_name} is not a masked array.')

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
        # Get time dimension index
        var_dims = self.get_variable_dimensions(var_name)
        time_dim_idx = var_dims.index(self._time_dim_name)

        # Get variable
        nc_var = self.second_data_file.variables[var_name]
        var = self._get_time_slice(nc_var, time_dim_idx, self.tidx_second)

        if np.ma.isMaskedArray(var):
            return np.ascontiguousarray(var.mask)

        raise PyLagRuntimeError(f'Variable {var_name} is not a masked array.')

    def _get_time_slice(self, nc_var, time_dim_idx: int, time_idx: int):
        """ Get the variable at the specified time index
        
        Parameters
        ----------
        nc_var : NetCDF4.Variable
            The NetCDF4 variable
        
        time_dim_idx : int
            The time dimension index
        
        time_idx : int
            The time index
        
        Returns
        -------
         : NDArray
             The variable array
        """
        n_dims = len(nc_var.shape)

        if n_dims == 1:
            return nc_var[time_idx]
        elif n_dims == 2:
            if time_dim_idx == 0:
                return nc_var[time_idx, :]
            elif time_dim_idx == 1:
                return nc_var[:, time_idx]
        elif n_dims == 3:
            if time_dim_idx == 0:
                return nc_var[time_idx, :, :]
            elif time_dim_idx == 1:
                return nc_var[:, time_idx, :]
            elif time_dim_idx == 2:
                return nc_var[:, :, time_idx]
        elif n_dims == 4:
            if time_dim_idx == 0:
                return nc_var[time_idx, :, :, :]
            elif time_dim_idx == 1:
                return nc_var[:, time_idx, :, :]
            elif time_dim_idx == 2:
                return nc_var[:, :, time_idx, :]
            elif time_dim_idx == 3:
                return nc_var[:, :, :, time_idx]
        else:
            raise PyLagRuntimeError('Variable has more than 4 dimensions - such variables '
                                    'are not supported.')

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
            self.first_data_file = self.dataset_reader.read_dataset(
                self.first_data_file_name)
            logger.info(f'Opened first data file {self.first_data_file_name} '
                        f'for reading.')
        except RuntimeError:
            logger.error(f'Could not open data file '
                         f'{self.first_data_file_name}.')
            raise PyLagRuntimeError('Could not open data file for reading.')

    def _open_second_data_file_for_reading(self):
        logger = logging.getLogger(__name__)

        # Close the second data file if one has been opened previously
        if self.second_data_file:
            self.second_data_file.close()

        # Open the second data file
        try:
            self.second_data_file = self.dataset_reader.read_dataset(
                self.second_data_file_name)
            logger.info(f'Opened second data file {self.second_data_file_name} '
                        f'for reading.')
        except RuntimeError:
            logger.error(f'Could not open data file '
                         f'{self.second_data_file_name}.')
            raise PyLagRuntimeError('Could not open data file for reading.')

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
            first_time_seconds.append((time -
                self.sim_start_datetime).total_seconds())

        self.first_time = np.array(first_time_seconds, dtype=DTYPE_FLOAT)

    def _set_second_time_array(self):
        # Second time array
        # ----------------
        second_datetime = self.datetime_reader.get_datetime(
            self.second_data_file)

        # Convert to seconds using datetime_start as a reference point
        second_time_seconds = []
        for time in second_datetime:
            second_time_seconds.append((time -
                self.sim_start_datetime).total_seconds())

        self.second_time = np.array(second_time_seconds, dtype=DTYPE_FLOAT)

    def _compute_first_dataset_time_delta(self, idx):
        # Time delta between two time points in the first time array
        # ----------------------------------------------------------
        if idx < len(self.first_time) - 1:
            return self.first_time[idx+1] - self.first_time[idx]
        else:
            return self.compute_time_delta_between_datasets(
                self.first_data_file_name, forward=True)

    def _compute_second_dataset_time_delta(self, idx):
        # Time delta between two time points in the second time array
        # -----------------------------------------------------------
        if idx > 0:
            return self.second_time[idx] - self.second_time[idx-1]
        else:
            return self.compute_time_delta_between_datasets(
                self.second_data_file_name, forward=False)

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
            logger.info(f'The provided time {time}s lies outside of the '
                        f'range for which there exists input data: '
                        f'{self.first_time[0]} to {self.first_time[-1]}s')
            raise PyLagValueError('Time out of range.')

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
            logger.info(f'The provided time {time}s lies outside of the range '
                        f'for which there exists input data: '
                        f'{self.second_time[0]} to {self.second_time[-1]}s')
            raise PyLagValueError('Time out of range.')
        
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
        return natsort.natsorted(glob.glob(f'{file_dir}/{file_name_stem}*.nc'))
                

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

        set_auto_maskandscale : bool
            Flag for masking

        Returns
        -------
         : NetCDF4 Dataset
            A NetCDF4 dataset.

        """
        ds = Dataset(file_name, 'r')
        ds.set_auto_maskandscale(set_auto_maskandscale)
        return ds


__all__ = ["FileReader",
           "FileNameReader",
           "DiskFileNameReader",
           "DatasetReader",
           "NetCDFDatasetReader"]
