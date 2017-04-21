from unittest import TestCase
import numpy.testing as test
import numpy as np
import datetime
from ConfigParser import SafeConfigParser

from pylag.gotm_data_reader import GOTMDataReader
from pylag import cwrappers

from pylag.mediator import Mediator

class MockGOTMMediator(Mediator):
    """ Test mediator for GOTM
    
    """
    def __init__(self):
        # Number of z levels
        n_zlay = 3
        self._dim_vars = {'z': n_zlay}
        
        # Last and next time points in seconds
        self._t_last = 0.0
        self._t_next = 1.0
        
        # Dictionaries holding the value of time dependent and time independent variables
        zeta = np.array([[0.0]])
        h = np.array([[[1.0, 2.0, 1.0]]]).reshape(n_zlay, 1, 1)
        z = np.array([[[-3.5, -2.0, -0.5]]]).reshape(n_zlay, 1, 1)
        nuh_last = np.array([[[1.0, 1.0, 0.0]]]).reshape(n_zlay, 1, 1)
        nuh_next = np.array([[[2.0, 2.0, 0.0]]]).reshape(n_zlay, 1, 1)
        self._time_dep_vars_last = {'zeta': zeta, 'h': h, 'z': z, 'nuh': nuh_last}
        self._time_dep_vars_next = {'zeta': zeta, 'h': h, 'z': z, 'nuh': nuh_next}
    
    def get_dimension_variable(self, var_name):
        return self._dim_vars[var_name]

    def get_time_at_last_time_index(self):
        return self._t_last

    def get_time_at_next_time_index(self):
        return self._t_next

    def get_time_dependent_variable_at_last_time_index(self, var_name, var_dims, var_type):
        return self._time_dep_vars_last[var_name][:].astype(var_type)

    def get_time_dependent_variable_at_next_time_index(self, var_name, var_dims, var_type):
        return self._time_dep_vars_next[var_name][:].astype(var_type)
    
class GOTMDataReader_test(TestCase):

    def setUp(self):
        # Create config
        config = SafeConfigParser()
        
        # Create mediator
        mediator = MockGOTMMediator()
        
        # Create data reader
        self.data_reader = GOTMDataReader(config, mediator)
        
        # Read in data
        datetime_start = datetime.datetime(2010,1,1) # Arbitrary start time
        datetime_end = datetime.datetime(2010,1,1) # Arbitrary end time
        self.data_reader.setup_data_access(datetime_start, datetime_end)

    def tearDown(self):
        del(self.data_reader)

    def test_get_zmin(self):
        xpos = 0.0
        ypos = 0.0
        host = 0

        time = 0.0
        self.data_reader.read_data(time)
        zmin = cwrappers.get_zmin(self.data_reader, time, xpos, ypos, host)
        test.assert_almost_equal(zmin, -4.0)

    def test_get_zmax(self):
        xpos = 0.0
        ypos = 0.0
        host = 0

        time = 0.0
        self.data_reader.read_data(time)
        zeta = cwrappers.get_zmax(self.data_reader, time, xpos, ypos, host)
        test.assert_almost_equal(zeta, 0.0)
 
        time = 0.5
        self.data_reader.read_data(time)
        zeta = cwrappers.get_zmax(self.data_reader, time, xpos, ypos, host)
        test.assert_almost_equal(zeta, 0.0)

        time = 1.0
        self.data_reader.read_data(time)
        zeta = cwrappers.get_zmax(self.data_reader, time, xpos, ypos, host)
        test.assert_almost_equal(zeta, 0.0)

    def test_set_vertical_grid_vars_for_a_particle_on_the_sea_surface(self):
        time = 0.0
        xpos = 0.0
        ypos = 0.0
        zpos = 0.0
        host = 0

        self.data_reader.read_data(time)
        grid_vars = cwrappers.set_vertical_grid_vars(self.data_reader, time, xpos, ypos, zpos, host)
        
        test.assert_equal(grid_vars['k_layer'], 2)
        test.assert_almost_equal(grid_vars['omega_interfaces'], 1.0)

    def test_set_vertical_grid_vars_for_a_particle_on_the_sea_floor(self):
        time = 0.0
        xpos = 0.0
        ypos = 0.0
        zpos = -4.0
        host = 0

        self.data_reader.read_data(time)
        grid_vars = cwrappers.set_vertical_grid_vars(self.data_reader, time, xpos, ypos, zpos, host)
        
        test.assert_equal(grid_vars['k_layer'], 0)
        test.assert_almost_equal(grid_vars['omega_interfaces'], 0.0)


    def test_set_vertical_grid_vars_for_a_particle_in_the_surface_boundary_layer(self):
        time = 0.0
        xpos = 0.0
        ypos = 0.0
        zpos = -0.5
        host = 0

        self.data_reader.read_data(time)
        grid_vars = cwrappers.set_vertical_grid_vars(self.data_reader, time, xpos, ypos, zpos, host)
        
        test.assert_equal(grid_vars['k_layer'], 2)
        test.assert_almost_equal(grid_vars['omega_interfaces'], 0.5)

    def test_set_vertical_grid_vars_for_a_particle_in_the_bottom_boundary_layer(self):
        time = 0.0
        xpos = 0.0
        ypos = 0.0
        zpos = -3.5
        host = 0

        self.data_reader.read_data(time)
        grid_vars = cwrappers.set_vertical_grid_vars(self.data_reader, time, xpos, ypos, zpos, host)
        
        test.assert_equal(grid_vars['k_layer'], 0)
        test.assert_almost_equal(grid_vars['omega_interfaces'], 0.5)

    def test_set_vertical_grid_vars_for_a_particle_in_the_middle_of_the_water_column(self):
        time = 0.0
        xpos = 0.0
        ypos = 0.0
        zpos = -2.0
        host = 0

        self.data_reader.read_data(time)
        grid_vars = cwrappers.set_vertical_grid_vars(self.data_reader, time, xpos, ypos, zpos, host)
        
        test.assert_equal(grid_vars['k_layer'], 1)
        test.assert_almost_equal(grid_vars['omega_interfaces'], 0.5)


    def test_get_vertical_eddy_diffusivity(self):
        xpos = 0.0
        ypos = 0.0
        host = 0

        time = 0.0
        zpos = -2.0
        self.data_reader.read_data(time)
        diffusivity = cwrappers.get_vertical_eddy_diffusivity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_almost_equal(diffusivity,  1.0)

        time = 0.5
        zpos = -2.0
        self.data_reader.read_data(time)
        diffusivity = cwrappers.get_vertical_eddy_diffusivity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_almost_equal(diffusivity,  1.5)

        time = 1.0
        zpos = -2.0
        self.data_reader.read_data(time)
        diffusivity = cwrappers.get_vertical_eddy_diffusivity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_almost_equal(diffusivity,  2.0)


    def test_get_vertical_eddy_diffusivity_derivative(self):
        xpos = 0.0
        ypos = 0.0
        host = 0

        time = 0.0
        zpos = -2.0
        self.data_reader.read_data(time)
        diffusivity_gradient = cwrappers.get_vertical_eddy_diffusivity_derivative(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_almost_equal(diffusivity_gradient, 0.0)

        time = 0.0
        zpos = -0.5
        self.data_reader.read_data(time)
        diffusivity_gradient = cwrappers.get_vertical_eddy_diffusivity_derivative(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_almost_equal(diffusivity_gradient, -1.0)

        time = 0.0
        zpos = 0.0
        self.data_reader.read_data(time)
        diffusivity_gradient = cwrappers.get_vertical_eddy_diffusivity_derivative(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_almost_equal(diffusivity_gradient, -1.0)
        