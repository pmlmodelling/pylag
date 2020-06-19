from unittest import TestCase
import numpy.testing as test
import numpy as np
import datetime

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.file_reader import FileReader, FileNameReader, DatasetReader

# Module level variables used in testing
# --------------------------------------

# Test datasets ('file_name': 'time variable array in seconds')
# NB The repition of 300 s is deliberate, and is introduced
# to test the behaviour when working with FVCOM data files where
# typically the last time in the first data file is repeated as
# the first entry in the second data file.
test_datasets = {'grid_metrics': [0.0],
                 'test_file_1': [0.0, 60.0, 120.0],
                 'test_file_2': [180.0, 240., 300.],
                 'test_file_3': [300., 360., 420., 480.]}

# Rounding interval in seconds
rounding_interval = "60"

# Units used with time variables
time_units = "seconds since 2000-01-01 00:00:00"


# Classes designed to mimic behaviours found in the NetCDF4 library
# -----------------------------------------------------------------

class TestVariable(list):
    """ Simple test variable class
    
    TestVariable is desgined to have similar behaviour to NetCDF4 Variable
    objects with respect to a) returning a data array when invoked, and b)
    having a units attribute. Introduced here to assist in testing.
    """
    def __init__(self, data, units):
        super(TestVariable, self).__init__(data)
        self.units = units


class TestDataset(object):
    """ Simple test dataset class
    
    TestDataset's are desgined to have similar behaviour to NetCDF4 Dataset
    objects. Introduced here to assist in testing.
    """
    def __init__(self, time_array):
        time_var = TestVariable(time_array, time_units)

        # Initialise variable dictionary
        self.variables = {'time': time_var}

    def close(self):
        del(self.variables)


# Mock objects which mimic the behaviour of accessing data on disk
# ----------------------------------------------------------------

class MockFileNameReader(FileNameReader):
    """ Mock file name reader
   
    Rather than return a list of filenames read from disk, as is
    the default behaviour in PyLag, return a list of pre-defined
    file names that will assist in testing FileReader's behaviour.
    """
    def __init__(self):
        self._file_names = ['test_file_1', 'test_file_2', 'test_file_3']

    def get_file_names(self, file_dir, file_name_stem):
        return self._file_names


class MockDatasetReader(DatasetReader):
    """ Mock dataset reader

    Rather than return a list of filenames read from disk, as is
    the default behaviour in PyLag, return a list of pre-defined
    file names that will assist in testing FileReader's behaviour.
    """
    def read_dataset(self, file_name):
        time_array = test_datasets[file_name]

        return TestDataset(time_array)


# FileReader test class
# ---------------------

class FileReader_test(TestCase):

    def setUp(self):
        # Create config
        self.config = configparser.SafeConfigParser()
        self.config.add_section("OCEAN_CIRCULATION_MODEL")
        self.config.set('OCEAN_CIRCULATION_MODEL', 'name', 'TEST')
        self.config.set('OCEAN_CIRCULATION_MODEL', 'data_dir', '')
        self.config.set('OCEAN_CIRCULATION_MODEL', 'data_file_stem', '')
        self.config.set('OCEAN_CIRCULATION_MODEL', 'grid_metrics_file', 'grid_metrics')
        self.config.set('OCEAN_CIRCULATION_MODEL', 'rounding_interval', rounding_interval)
        self.config.add_section("SIMULATION")

        # Mock file name reader
        self.file_name_reader = MockFileNameReader()

        # Mock dataset reader
        self.dataset_reader = MockDatasetReader()

    def tearDown(self):
        del(self.config)
        del(self.file_name_reader)
        del(self.dataset_reader)

    def test_use_start_datetime_equal_to_data_record_start(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid - 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

    def test_use_end_datetime_equal_to_data_record_start(self):
        # Should be valid during reverse tracking
        start_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 second before data record end
        end_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid - 0 seconds after data record start

        self.config.set('SIMULATION', 'time_direction', 'reverse')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

    def test_use_start_datetime_equal_to_data_record_end(self):
        start_datetime = datetime.datetime(2000,1,1,0,8,0) # Invalid - 0 seconds after data record end
        end_datetime = datetime.datetime(2000,1,1,0,0,0) # Invalid - 0 seconds after data record start

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.assertRaises(ValueError, FileReader, self.config, self.file_name_reader, self.dataset_reader,
                          start_datetime, end_datetime)

    def test_use_end_datetime_equal_to_data_record_end(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid - 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,8,0) # Invalid - 0 seconds after data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.assertRaises(ValueError, FileReader, self.config, self.file_name_reader, self.dataset_reader,
                          start_datetime, end_datetime)

    def test_use_start_datetime_before_data_record_start(self):
        start_datetime = datetime.datetime(1999,1,1,0,0,0) # Invalid - 1 year before data start date
        end_datetime = datetime.datetime(2000,1,1,0,8,0) # Valid - 480 seconds

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.assertRaises(ValueError, FileReader, self.config, self.file_name_reader, self.dataset_reader,
                          start_datetime, end_datetime)

    def test_use_start_datetime_after_data_record_end(self):
        start_datetime = datetime.datetime(2001,1,1,8,0) # Invalid - 1 year after data end date
        end_datetime = datetime.datetime(2000,1,1,0,8,0) # Valid - 480 seconds

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.assertRaises(ValueError, FileReader, self.config, self.file_name_reader, self.dataset_reader,
                          start_datetime, end_datetime)

    def test_use_end_datetime_before_data_record_start(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid - 0 seconds after data record start
        end_datetime = datetime.datetime(1999,1,1,0,0,0) # Invalid - 1 year before data start date

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.assertRaises(ValueError, FileReader, self.config, self.file_name_reader, self.dataset_reader,
                          start_datetime, end_datetime)

    def test_use_end_datetime_after_data_record_start(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid - 0 seconds after data record start
        end_datetime = datetime.datetime(2001,1,1,0,8,0) # Invalid - 1 year after data end date

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.assertRaises(ValueError, FileReader, self.config, self.file_name_reader, self.dataset_reader,
                          start_datetime, end_datetime)

    def test_set_file_names_with_start_datetime_equal_to_data_record_start(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid - 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check file names
        test.assert_array_equal('test_file_1', self.file_reader.first_data_file_name) 
        test.assert_array_equal('test_file_1', self.file_reader.second_data_file_name) 

    def test_set_file_names_with_start_datetime_in_the_first_data_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,1,0) # Valid = 60 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check file names
        test.assert_array_equal('test_file_1', self.file_reader.first_data_file_name) 
        test.assert_array_equal('test_file_1', self.file_reader.second_data_file_name) 

    def test_set_file_names_with_start_datetime_equal_to_the_last_time_point_in_the_first_data_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,2,0) # Valid = 120 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check file names
        test.assert_array_equal('test_file_1', self.file_reader.first_data_file_name) 
        test.assert_array_equal('test_file_2', self.file_reader.second_data_file_name) 

    def test_set_file_names_with_start_datetime_inbetween_two_data_files(self):
        start_datetime = datetime.datetime(2000,1,1,0,2,30) # Valid = 150 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check file names
        test.assert_array_equal('test_file_1', self.file_reader.first_data_file_name) 
        test.assert_array_equal('test_file_2', self.file_reader.second_data_file_name) 

    def test_set_file_names_with_start_datetime_in_the_second_data_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,4,0) # Valid = 240 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check file names
        test.assert_array_equal('test_file_2', self.file_reader.first_data_file_name) 
        test.assert_array_equal('test_file_2', self.file_reader.second_data_file_name) 

    def test_set_file_names_with_time_points_repeated_between_adjacent_files(self):
        start_datetime = datetime.datetime(2000,1,1,0,5,0) # Valid = 300 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check file names
        test.assert_array_equal('test_file_2', self.file_reader.first_data_file_name)
        test.assert_array_equal('test_file_3', self.file_reader.second_data_file_name)

    def test_set_time_arrays_with_start_datetime_equal_to_data_record_start(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid = 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check time arrays
        test.assert_array_almost_equal([0., 60., 120.], self.file_reader.first_time)
        test.assert_array_almost_equal([0., 60., 120.], self.file_reader.second_time)

    def test_set_time_arrays_with_start_datetime_equal_to_the_last_time_point_in_the_first_data_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,2,0) # Valid = 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check time arrays
        test.assert_array_almost_equal([-120., -60., 0.], self.file_reader.first_time)
        test.assert_array_almost_equal([60., 120., 180.], self.file_reader.second_time)

    def test_set_time_indices_with_start_datetime_equal_to_data_record_start(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid = 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check time indices
        test.assert_equal(0, self.file_reader.tidx_first)
        test.assert_equal(1, self.file_reader.tidx_second)

    def test_set_time_indices_with_start_datetime_equal_to_the_last_time_point_in_the_first_data_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,2,0) # Valid = 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check time indices
        test.assert_equal(2, self.file_reader.tidx_first)
        test.assert_equal(0, self.file_reader.tidx_second)

    def test_set_time_indices_with_time_points_repeated_between_adjacent_files(self):
        start_datetime = datetime.datetime(2000,1,1,0,5,0) # Valid = 300 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check time indices
        test.assert_equal(2, self.file_reader.tidx_first)
        test.assert_equal(1, self.file_reader.tidx_second)

    def test_set_file_names_when_updating_reading_frames_in_the_first_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid = 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Update reading frames
        self.file_reader.update_reading_frames(time=60)

        # Check file names
        test.assert_array_equal('test_file_1', self.file_reader.first_data_file_name)
        test.assert_array_equal('test_file_1', self.file_reader.second_data_file_name)

    def test_set_time_indices_when_updating_reading_frames_in_the_first_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid = 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Update reading frames
        self.file_reader.update_reading_frames(time=60)

        # Check time indices
        test.assert_equal(1, self.file_reader.tidx_first)
        test.assert_equal(2, self.file_reader.tidx_second)

    def test_set_file_names_when_updating_reading_frames_at_the_end_of_the_first_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid = 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Update reading frames
        self.file_reader.update_reading_frames(time=120.)

        # Check file names
        test.assert_array_equal('test_file_1', self.file_reader.first_data_file_name)
        test.assert_array_equal('test_file_2', self.file_reader.second_data_file_name)

    def test_set_time_indices_when_updating_reading_frames_at_the_end_of_the_first_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid = 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Update reading frames
        self.file_reader.update_reading_frames(time=120.)

        # Check time indices
        test.assert_equal(2, self.file_reader.tidx_first)
        test.assert_equal(0, self.file_reader.tidx_second)

    def test_set_file_names_when_updating_reading_frames_after_the_end_of_the_first_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid = 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Update reading frames
        self.file_reader.update_reading_frames(time=150.)

        # Check file names
        test.assert_array_equal('test_file_1', self.file_reader.first_data_file_name)
        test.assert_array_equal('test_file_2', self.file_reader.second_data_file_name)

    def test_set_time_indices_when_updating_reading_frames_after_the_end_of_the_first_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid = 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Update reading frames
        self.file_reader.update_reading_frames(time=150.)

        # Check time indices
        test.assert_equal(2, self.file_reader.tidx_first)
        test.assert_equal(0, self.file_reader.tidx_second)

    def test_set_file_names_when_updating_reading_frames_at_the_start_of_the_second_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid = 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Update reading frames
        self.file_reader.update_reading_frames(time=180.)

        # Check file names
        test.assert_array_equal('test_file_2', self.file_reader.first_data_file_name)
        test.assert_array_equal('test_file_2', self.file_reader.second_data_file_name)

    def test_set_time_indices_when_updating_reading_frames_at_the_start_of_the_second_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid = 0 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,7,59) # Valid - 1 seconds before data record end

        self.config.set('SIMULATION', 'time_direction', 'forward')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Update reading frames
        self.file_reader.update_reading_frames(time=180.)

        # Check time indices
        test.assert_equal(0, self.file_reader.tidx_first)
        test.assert_equal(1, self.file_reader.tidx_second)

    def test_set_file_names_when_reverse_tracking(self):
        start_datetime = datetime.datetime(2000,1,1,0,3,0) # Valid = 180 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid - 0 seconds before data record start

        self.config.set('SIMULATION', 'time_direction', 'reverse')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check file names
        test.assert_array_equal('test_file_1', self.file_reader.first_data_file_name)
        test.assert_array_equal('test_file_2', self.file_reader.second_data_file_name)

    def test_set_file_names_when_updating_reading_frames_during_reverse_tracking_between_data_files(self):
        start_datetime = datetime.datetime(2000,1,1,0,3,0) # Valid = 180 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid - 0 seconds before data record start

        self.config.set('SIMULATION', 'time_direction', 'reverse')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Update reading frames
        self.file_reader.update_reading_frames(time=-60.)

        # Check file names
        test.assert_array_equal('test_file_1', self.file_reader.first_data_file_name)
        test.assert_array_equal('test_file_1', self.file_reader.second_data_file_name)

    def test_set_time_indices_when_reverse_tracking(self):
        start_datetime = datetime.datetime(2000,1,1,0,3,0) # Valid = 180 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid - 0 seconds before data record start

        self.config.set('SIMULATION', 'time_direction', 'reverse')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Check time indices
        test.assert_equal(2, self.file_reader.tidx_first)
        test.assert_equal(0, self.file_reader.tidx_second)

    def test_set_time_indices_when_updating_reading_frames_during_reverse_tracking_between_data_files(self):
        start_datetime = datetime.datetime(2000,1,1,0,3,0) # Valid = 180 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid - 0 seconds before data record start

        self.config.set('SIMULATION', 'time_direction', 'reverse')

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Update reading frames
        self.file_reader.update_reading_frames(time=-60.)

        # Check time indices
        test.assert_equal(1, self.file_reader.tidx_first)
        test.assert_equal(2, self.file_reader.tidx_second)

    def test_set_file_names_when_updating_reading_frames_during_reverse_tracking_into_first_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,3,0) # Valid = 180 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid - 0 seconds before data record start

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Update reading frames
        self.file_reader.update_reading_frames(time=-90.)

        # Check file names
        test.assert_array_equal('test_file_1', self.file_reader.first_data_file_name)
        test.assert_array_equal('test_file_1', self.file_reader.second_data_file_name)

    def test_set_time_indices_when_updating_reading_frames_during_reverse_tracking_into_first_file(self):
        start_datetime = datetime.datetime(2000,1,1,0,3,0) # Valid = 180 seconds after data record start
        end_datetime = datetime.datetime(2000,1,1,0,0,0) # Valid - 0 seconds before data record start

        # Create file reader
        self.file_reader = FileReader(self.config, self.file_name_reader, self.dataset_reader, start_datetime, end_datetime)

        # Update reading frames
        self.file_reader.update_reading_frames(time=-90.)

        # Check time indices
        test.assert_equal(1, self.file_reader.tidx_first)
        test.assert_equal(2, self.file_reader.tidx_second)


