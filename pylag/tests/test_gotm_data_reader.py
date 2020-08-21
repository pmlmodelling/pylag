from unittest import TestCase
import numpy.testing as test
import numpy as np
import datetime

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.gotm_data_reader import GOTMDataReader
from pylag.boundary_conditions import RefVertBoundaryConditionCalculator
from pylag.particle_cpp_wrapper import ParticleSmartPtr

from pylag.mediator import Mediator

class MockGOTMMediator(Mediator):
    """ Test mediator for GOTM
    
    """
    def __init__(self):
        # Number of z layers
        n_zlay = 3
        
        # Numer of z levels (i.e. layer interfaces)
        n_zlev = 4
        
        self._dim_vars = {'z': n_zlay, 'zi': n_zlev}
        
        # Last and next time points in seconds
        self._t_last = 0.0
        self._t_next = 1.0
        
        # Dictionaries holding the value of time dependent and time independent variables
        zeta = np.array([[0.0]])
        z = np.array([[[-3.5, -2.0, -0.5]]]).reshape(n_zlay, 1, 1)
        zi = np.array([[[-4.0, -3.0, -1.0, 0.0]]]).reshape(n_zlev, 1, 1)
        nuh_last = np.array([[[0.0, 1.0, 1.0, 0.0]]]).reshape(n_zlev, 1, 1)
        nuh_next = np.array([[[0.0, 2.0, 2.0, 0.0]]]).reshape(n_zlev, 1, 1)
        self._time_dep_vars_last = {'zeta': zeta, 'z': z, 'zi': zi, 'nuh': nuh_last}
        self._time_dep_vars_next = {'zeta': zeta, 'z': z, 'zi': zi, 'nuh': nuh_next}

    def setup_data_access(self, start_datetime, end_datetime):
        pass

    def update_reading_frames(self, time):
        pass
    
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
        config = configparser.ConfigParser()
        config.add_section("SIMULATION")
        config.set('SIMULATION', 'time_direction', 'forward')
        config.add_section("OCEAN_CIRCULATION_MODEL")
        config.set('OCEAN_CIRCULATION_MODEL', 'vertical_interpolation_scheme', 'linear')
        
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
        x1 = 0.0
        x2 = 0.0
        

        time = 0.0
        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2)
        bathy = self.data_reader.get_zmin_wrapper(time, particle)
        test.assert_almost_equal(bathy, -4.0)

    def test_get_zmax(self):
        x1 = 0.0
        x2 = 0.0
        

        time = 0.0
        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2)
        zeta = self.data_reader.get_zmax_wrapper(time, particle)
        test.assert_almost_equal(zeta, 0.0)
 
        time = 0.5
        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2)
        zeta = self.data_reader.get_zmax_wrapper(time, particle)
        test.assert_almost_equal(zeta, 0.0)

        time = 1.0
        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2)
        zeta = self.data_reader.get_zmax_wrapper(time, particle)
        test.assert_almost_equal(zeta, 0.0)

    def test_set_vertical_grid_vars_for_a_particle_on_the_sea_surface(self):
        time = 0.0
        x1 = 0.0
        x2 = 0.0
        x3 = 0.0
        

        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3)
        self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        
        test.assert_equal(particle.k_layer, 2)
        test.assert_almost_equal(particle.omega_interfaces, 1.0)

    def test_set_vertical_grid_vars_for_a_particle_on_the_sea_floor(self):
        time = 0.0
        x1 = 0.0
        x2 = 0.0
        x3 = -4.0
        

        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3)
        self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        
        test.assert_equal(particle.k_layer, 0)
        test.assert_almost_equal(particle.omega_interfaces, 0.0)


    def test_set_vertical_grid_vars_for_a_particle_in_the_surface_boundary_layer(self):
        time = 0.0
        x1 = 0.0
        x2 = 0.0
        x3 = -0.5
        

        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 2)
        test.assert_almost_equal(particle.omega_interfaces, 0.5)

    def test_set_vertical_grid_vars_for_a_particle_in_the_bottom_boundary_layer(self):
        time = 0.0
        x1 = 0.0
        x2 = 0.0
        x3 = -3.5
        

        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 0)
        test.assert_almost_equal(particle.omega_interfaces, 0.5)

    def test_set_vertical_grid_vars_for_a_particle_in_the_middle_of_the_water_column(self):
        time = 0.0
        x1 = 0.0
        x2 = 0.0
        x3 = -2.0
        

        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 1)
        test.assert_almost_equal(particle.omega_interfaces, 0.5)

    def test_get_vertical_eddy_diffusivity(self):
        x1 = 0.0
        x2 = 0.0
        

        time = 0.0
        x3 = -2.0
        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        diffusivity = self.data_reader.get_vertical_eddy_diffusivity_wrapper(time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(diffusivity,  1.0)

        time = 0.5
        x3 = -2.0
        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        diffusivity = self.data_reader.get_vertical_eddy_diffusivity_wrapper(time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(diffusivity,  1.5)

        time = 1.0
        x3 = -2.0
        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        diffusivity = self.data_reader.get_vertical_eddy_diffusivity_wrapper(time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(diffusivity,  2.0)


    def test_get_vertical_eddy_diffusivity_derivative(self):
        x1 = 0.0
        x2 = 0.0
        

        time = 0.0
        x3 = -2.0
        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        diffusivity_gradient = self.data_reader.get_vertical_eddy_diffusivity_derivative_wrapper(time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(diffusivity_gradient, 0.0)

        time = 0.0
        x3 = 0.0
        self.data_reader.read_data(time)
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        diffusivity_gradient = self.data_reader.get_vertical_eddy_diffusivity_derivative_wrapper(time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(diffusivity_gradient, -1.0)


class GOTMReflectingVertBoundaryCondition_test(TestCase):

    def setUp(self):
        # Create config
        config = configparser.ConfigParser()
        config.add_section("SIMULATION")
        config.set('SIMULATION', 'time_direction', 'forward')
        config.add_section("OCEAN_CIRCULATION_MODEL")
        config.set('OCEAN_CIRCULATION_MODEL', 'vertical_interpolation_scheme', 'linear')
        
        # Create mediator
        mediator = MockGOTMMediator()
        
        # Create data reader
        self.data_reader = GOTMDataReader(config, mediator)
        
        # Read in data
        datetime_start = datetime.datetime(2010,1,1) # Arbitrary start time
        datetime_end = datetime.datetime(2010,1,1) # Arbitrary end time
        self.data_reader.setup_data_access(datetime_start, datetime_end)

        # Boundary condition calculator
        self.vert_boundary_condition_calculator = RefVertBoundaryConditionCalculator()

    def tearDown(self):
        del(self.data_reader)


    def test_apply_reflecting_boundary_condition_for_a_particle_that_has_pierced_the_free_surface(self):
        x3 = 0.0 + 0.1
        

        time = 0.0
        
        particle = ParticleSmartPtr(x3=x3)
        self.data_reader.read_data(time)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.vert_boundary_condition_calculator.apply_wrapper(self.data_reader, time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(particle.x3, -0.1)

    def test_apply_reflecting_boundary_condition_for_a_particle_that_has_just_pierced_the_free_surface(self):
        x3 = 0.0 + 1.e-15
        

        time = 0.0
        
        particle = ParticleSmartPtr(x3=x3)
        self.data_reader.read_data(time)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.vert_boundary_condition_calculator.apply_wrapper(self.data_reader, time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(particle.x3, 0.0)

    def test_apply_reflecting_boundary_condition_for_a_particle_that_has_pierced_the_sea_floor(self):
        x3 = -4.0 - 0.1
        

        time = 0.0
        
        particle = ParticleSmartPtr(x3=x3)
        self.data_reader.read_data(time)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.vert_boundary_condition_calculator.apply_wrapper(self.data_reader, time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(particle.x3, -3.9)

    def test_apply_reflecting_boundary_condition_for_a_particle_that_has_just_pierced_the_free_surface(self):
        x3 = -4.0 - 1.e-15
        

        time = 0.0
        
        particle = ParticleSmartPtr(x3=x3)
        self.data_reader.read_data(time)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.vert_boundary_condition_calculator.apply_wrapper(self.data_reader, time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(particle.x3, -4.0)

