from unittest import TestCase
import numpy.testing as test
from ConfigParser import SafeConfigParser

from pylag.fvcom_data_reader import FVCOMDataReader
from pylag import cwrappers
        
class FVCOMDataReader_test(TestCase):

    def setUp(self):
        config = SafeConfigParser()
        config.add_section("SIMULATION")
        config.add_section("OCEAN_CIRCULATION_MODEL")
        config.set('SIMULATION', 'start_datetime', '2013-01-06 00:00:00')
        config.set('OCEAN_CIRCULATION_MODEL', 'data_dir', '../resources/')
        config.set('OCEAN_CIRCULATION_MODEL', 'grid_metrics_file', 'fvcom_grid_metrics_test.nc')
        config.set('OCEAN_CIRCULATION_MODEL', 'data_file_stem', 'fvcom_data_test')
        config.set('OCEAN_CIRCULATION_MODEL', 'rounding_interval', '3600')
        self.data_reader = FVCOMDataReader(config)

    def tearDown(self):
        del(self.data_reader)

    def test_find_host_using_local_search(self):
        xpos = 365751.7
        ypos = 5323568.0
        guess = 1 # Known neighbour
        host = self.data_reader.find_host_using_local_search(xpos, ypos, guess)
        test.assert_equal(host, 0)
    
    def test_find_host_using_global_search(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = self.data_reader.find_host_using_global_search(xpos, ypos)
        test.assert_equal(host, 0)

    def test_get_bathymetry(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0
        bathy = self.data_reader.get_bathymetry(xpos, ypos, host)
        test.assert_almost_equal(bathy, 11.0)

    def test_get_sea_sur_elev(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0
        
        time = 0.0
        zeta = self.data_reader.get_sea_sur_elev(time, xpos, ypos, host)
        test.assert_almost_equal(zeta, 1.0)
        
        time = 1800.0
        zeta = self.data_reader.get_sea_sur_elev(time, xpos, ypos, host)
        test.assert_almost_equal(zeta, 1.5)

    def test_get_velocity_in_surface_layer(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0

        zpos = 0.0
        time = 0.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_array_almost_equal(vel, [2.0, 2.0, 0.0])

        zpos = 0.0
        time = 1800.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_array_almost_equal(vel, [3.0, 3.0, 0.0])
        
        zpos = -0.1
        time = 0.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_array_almost_equal(vel, [2.0, 2.0, 0.5])

    def test_get_velocity_in_middle_layer(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0

        zpos = -0.3
        time = 0.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_array_almost_equal(vel, [1.5, 1.5, 1.0])

        zpos = -0.3
        time = 1800.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_array_almost_equal(vel, [2.25, 2.25, 1.5])

    def test_get_velocity_in_bottom_layer(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0

        zpos = -1.0
        time = 0.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_array_almost_equal(vel, [0.0, 0.0, 0.0])

        zpos = -1.0
        time = 1800.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_array_almost_equal(vel, [0.0, 0.0, 0.0])
        
        zpos = -0.9
        time = 0.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_array_almost_equal(vel, [0.0, 0.0, 0.5])