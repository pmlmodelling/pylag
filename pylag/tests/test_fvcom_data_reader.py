from unittest import TestCase
import numpy.testing as test
import numpy as np
import datetime

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

# PyLag data types
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT

from pylag.fvcom_data_reader import FVCOMDataReader
from pylag.boundary_conditions import RefHorizBoundaryConditionCalculator
from pylag.boundary_conditions import RefVertBoundaryConditionCalculator
from pylag.particle_cpp_wrapper import ParticleSmartPtr
from pylag import cwrappers

from pylag.mediator import Mediator

class MockFVCOMMediator(Mediator):
    """ Test mediator for FVCOM

    """
    def __init__(self):
        # Dimension variables
        n_elems = 5
        n_nodes = 7
        n_siglev = 4
        n_siglay = 3
        self._dim_vars = {'element': n_elems, 'node': n_nodes,
                          'siglev': n_siglev, 'siglay': n_siglay}
                          
        # Grid variables
        nv = np.array([[4,0,2,4,3],[3,1,4,5,6],[1,3,1,3,0]],dtype=int)
        nbe = np.array([[1,0,0,-1,-1],[2,4,-1,0,1],[3,-1,-1,-1,-1]],dtype=int)
        x = np.array([2.0, 1.0, 0.0, 2.0, 1.0, 1.5, 3.0], dtype=float)
        y = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0], dtype=float)
        xc = np.array([1.3333333333, 1.6666666667, 0.6666666667, 1.5000000000, 2.3333333333], dtype=float)
        yc = np.array([1.6666666667, 1.3333333333, 1.3333333333, 2.3333333333, 1.3333333333], dtype=float)
        siglay = np.array([[-.1,-.5,-.9],]*n_nodes, dtype=float).transpose()
        siglev = np.array([[0., -.2,-.8,-1.],]*n_nodes, dtype=float).transpose()
        h = np.array([10., 11., 10., 11., 11., 10.], dtype=float)
        self._grid_vars = {'nv': nv, 'nbe': nbe, 'x': x, 'y': y, 'xc': xc,
                           'yc': yc, 'siglay': siglay, 'siglev': siglev,
                           'h': h}
        
        # Dictionaries holding the value of time dependent and time independent variables
        zeta = np.array([[0.,1.,0.,1.,1.,0.],[0.,2.,0.,2.,2.,0.]],dtype=float)
        
        # u/v/w are imposed, equal across elements, decreasing with depth and increasing in time
        uvw_t0 = np.array([[2.]*n_elems,[1.]*n_elems,[0.]*n_elems], dtype=float)
        uvw_t1 = np.array([[4.]*n_elems,[2.]*n_elems,[0.]*n_elems], dtype=float)
        u = np.array([uvw_t0, uvw_t1], dtype=float)
        v = np.array([uvw_t0, uvw_t1], dtype=float)
        ww = np.array([uvw_t0, uvw_t1], dtype=float)
        
        # kh is imposed, equal across nodes, variable with depth and increasing in time
        kh_t0 = np.array([[0.]*n_nodes,[0.01]*n_nodes,[0.01]*n_nodes,[0.]*n_nodes], dtype=float)
        kh_t1 = np.array([[0.]*n_nodes,[0.1]*n_nodes,[0.1]*n_nodes,[0.]*n_nodes], dtype=float)
        kh = np.array([kh_t0, kh_t1], dtype=float)

        # viscofh is imposed, equal across nodes, variable with depth and increasing in time
        viscofh_t0 = np.array([[0.01]*n_nodes,[0.01]*n_nodes,[0.]*n_nodes], dtype=float)
        viscofh_t1 = np.array([[0.1]*n_nodes,[0.1]*n_nodes,[0.]*n_nodes], dtype=float)
        viscofh = np.array([viscofh_t0, viscofh_t1], dtype=float)

        # thetao is imposed, equal across nodes, variable with depth and increasing in time
        thetao_t0 = np.array([[0.01]*n_nodes,[0.01]*n_nodes,[0.]*n_nodes], dtype=float)
        thetao_t1 = np.array([[0.1]*n_nodes,[0.1]*n_nodes,[0.]*n_nodes], dtype=float)
        thetao = np.array([thetao_t0, thetao_t1], dtype=float)

        # so is imposed, equal across nodes, variable with depth and increasing in time
        so_t0 = np.array([[0.01]*n_nodes,[0.01]*n_nodes,[0.]*n_nodes], dtype=float)
        so_t1 = np.array([[0.1]*n_nodes,[0.1]*n_nodes,[0.]*n_nodes], dtype=float)
        so = np.array([so_t0, so_t1], dtype=float)

        # Wet cells
        wet_cells = np.array([[1,1,1,1],[1,1,1,1]],dtype=int)

        # Store in dictionaries
        self._time_dep_vars_last = {'zeta': zeta[0,:], 'u': u[0,:], 'v': v[0,:],
                                    'ww': ww[0,:], 'kh': kh[0,:],
                                    'viscofh': viscofh[0,:],
                                    'temp': thetao[0,:], 'salinity': so[0,:],
                                    'wet_cells': wet_cells[0,:]}
        self._time_dep_vars_next = {'zeta': zeta[1,:], 'u': u[1,:], 'v': v[1,:],
                                    'ww': ww[1,:], 'kh': kh[1,:],
                                    'viscofh': viscofh[1,:],
                                    'temp': thetao[1,:], 'salinity': so[1,:],
                                    'wet_cells': wet_cells[1,:]}

        # Time in seconds. ie two time pts, 1 hour apart
        self._t_last = 0.0
        self._t_next = 3600.0

    def setup_data_access(self, start_datetime, end_datetime):
        pass

    def get_dimension_variable(self, var_name):
        return self._dim_vars[var_name]

    def get_time_at_last_time_index(self):
        return self._t_last

    def get_time_at_next_time_index(self):
        return self._t_next

    def get_grid_variable(self, var_name, var_dims, var_type):
        return self._grid_vars[var_name][:].astype(var_type)

    def get_time_dependent_variable_at_last_time_index(self, var_name, var_dims, var_type):
        return self._time_dep_vars_last[var_name][:].astype(var_type)

    def get_time_dependent_variable_at_next_time_index(self, var_name, var_dims, var_type):
        return self._time_dep_vars_next[var_name][:].astype(var_type)

class FVCOMDataReader_test(TestCase):

    def setUp(self):
        # Create config
        config = configparser.SafeConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')
        config.add_section("SIMULATION")
        config.set('SIMULATION', 'time_direction', 'forward')
        config.add_section("OCEAN_CIRCULATION_MODEL")
        config.set('OCEAN_CIRCULATION_MODEL', 'has_Kh', 'True')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_Ah', 'True')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_is_wet', 'True')
        config.set('OCEAN_CIRCULATION_MODEL', 'coordinate_system', 'cartesian')
        config.add_section("OUTPUT")
        config.set('OUTPUT', 'environmental_variables', 'thetao, so')
        # Create mediator
        mediator = MockFVCOMMediator()
        
        # Create data reader
        self.data_reader = FVCOMDataReader(config, mediator)
        
        # Read in data
        datetime_start = datetime.datetime(2000,1,1) # Arbitrary start time, ignored by mock mediator
        datetime_end = datetime.datetime(2000,1,1) # Arbitrary end time, ignored by mock mediator
        self.data_reader.setup_data_access(datetime_start, datetime_end)

        # Grid offsets
        self.xmin = self.data_reader.get_xmin()
        self.ymin = self.data_reader.get_ymin()

    def tearDown(self):
        del(self.data_reader)

    def test_find_host_using_global_search(self):
        particle = ParticleSmartPtr(x1=1.3333333333-self.xmin, x2=1.6666666667-self.ymin)
        flag = self.data_reader.find_host_using_global_search_wrapper(particle)
        test.assert_equal(particle.get_host_horizontal_elem('fvcom'), 0)
        test.assert_equal(flag, 0)

    def test_find_host_using_global_search_when_a_particle_is_in_an_element_with_two_land_boundaries(self):
        particle = ParticleSmartPtr(x1=0.6666666667-self.xmin, x2=1.3333333333-self.ymin, host_elements={'fvcom': -1})
        flag = self.data_reader.find_host_using_global_search_wrapper(particle)
        test.assert_equal(particle.get_host_horizontal_elem('fvcom'), -1)

    def test_find_host_when_particle_is_in_the_domain(self):
        particle_old = ParticleSmartPtr(x1=1.5-self.xmin, x2=2.3333333333-self.ymin, host_elements={'fvcom': 3})
        particle_new = ParticleSmartPtr(x1=1.3333333333-self.xmin, x2=1.6666666667-self.ymin, host_elements={'fvcom': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 0)

    def test_find_host_when_particle_has_crossed_a_land_boundary(self):
        particle_old = ParticleSmartPtr(x1=1.6666666667-self.xmin, x2=1.3333333333-self.ymin, host_elements={'fvcom': 1})
        particle_new = ParticleSmartPtr(x1=1.5-self.xmin, x2=0.0-self.ymin, host_elements={'fvcom': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, -1)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 1)

    def test_find_host_when_particle_has_crossed_into_an_element_with_two_land_boundaries(self):
        particle_old = ParticleSmartPtr(x1=1.3333333333-self.xmin, x2=1.6666666667-self.ymin, host_elements={'fvcom': 0})
        particle_new = ParticleSmartPtr(x1=0.6666666667-self.xmin, x2=1.3333333333-self.ymin, host_elements={'fvcom': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, -1)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 0)

    def test_find_host_when_particle_has_crossed_multiple_elements_to_an_element_with_two_land_boundaries(self):
        particle_old = ParticleSmartPtr(x1=1.6666666667-self.xmin, x2=1.3333333333-self.ymin, host_elements={'fvcom': 1})
        particle_new = ParticleSmartPtr(x1=0.6666666667-self.xmin, x2=1.3333333333-self.ymin, host_elements={'fvcom': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, -1)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 0)

    def test_find_host_when_particle_is_just_inside_an_internal_element(self):
        particle_old = ParticleSmartPtr(x1=1.4-self.xmin, x2=1.5-self.ymin, host_elements={'fvcom': 0})
        particle_new = ParticleSmartPtr(x1=1.5 - 1.e-15 - self.xmin, x2=1.5 - self.ymin, host_elements={'fvcom': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 0)

    def test_find_host_when_particle_is_just_outside_an_internal_element(self):
        particle_old = ParticleSmartPtr(x1=1.4-self.xmin, x2=1.5-self.ymin, host_elements={'fvcom': 0})
        particle_new = ParticleSmartPtr(x1=1.5 + 1.e-15 - self.xmin, x2=1.5 - self.ymin, host_elements={'fvcom': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 1)

    def test_find_host_when_particle_is_on_an_internal_elements_side(self):
        particle_old = ParticleSmartPtr(x1=1.4 - self.xmin, x2=1.5 - self.ymin, host_elements={'fvcom': 0})
        particle_new = ParticleSmartPtr(x1=1.5 - self.xmin, x2=1.5 - self.ymin, host_elements={'fvcom': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 0)

    def test_find_host_when_particle_is_just_inside_an_external_element(self):
        particle_old = ParticleSmartPtr(x1=1.6666666667-self.xmin, x2=1.3333333333-self.ymin, host_elements={'fvcom': 1})
        particle_new = ParticleSmartPtr(x1=1.6666666667-self.xmin, x2=1.0 + 1.e-15 - self.ymin, host_elements={'fvcom': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 1)

    def test_find_host_when_particle_is_just_outside_an_external_element(self):
        particle_old = ParticleSmartPtr(x1=1.6666666667-self.xmin, x2=1.3333333333-self.ymin, host_elements={'fvcom': 1})
        particle_new = ParticleSmartPtr(x1=1.6666666667-self.xmin, x2=1.0 - 1.e-15 - self.ymin, host_elements={'fvcom': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, -1)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 1)

    def test_find_host_when_particle_is_on_an_external_elements_side(self):
        particle_old = ParticleSmartPtr(x1=1.6666666667-self.xmin, x2=1.3333333333-self.ymin, host_elements={'fvcom': 1})
        particle_new = ParticleSmartPtr(x1=1.6666666667-self.xmin, x2=1.0-self.ymin, host_elements={'fvcom': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 1)

    def test_get_boundary_intersection(self):
        particle_old = ParticleSmartPtr(x1=1.6666666667-self.xmin, x2=1.3333333333-self.ymin, host_elements={'fvcom': 1})
        particle_new = ParticleSmartPtr(x1=1.6666666667-self.xmin, x2=0.9-self.ymin, host_elements={'fvcom': 1})
        intersection = self.data_reader.get_boundary_intersection_wrapper(particle_old, particle_new)
        test.assert_almost_equal(intersection.x1_py+self.xmin, 2.0)
        test.assert_almost_equal(intersection.y1_py+self.ymin, 1.0)
        test.assert_almost_equal(intersection.x2_py+self.xmin, 1.0)
        test.assert_almost_equal(intersection.y2_py+self.ymin, 1.0)
        test.assert_almost_equal(intersection.xi_py+self.xmin, 1.6666666667)
        test.assert_almost_equal(intersection.yi_py+self.ymin, 1.0)

    def test_get_boundary_intersection_when_a_particle_has_left_an_external_elements_edge(self):
        particle_old = ParticleSmartPtr(x1=1.5-self.xmin, x2=1.0-self.ymin, host_elements={'fvcom': 1})
        particle_new = ParticleSmartPtr(x1=1.5-self.xmin, x2=0.9-self.ymin, host_elements={'fvcom': 1})
        intersection = self.data_reader.get_boundary_intersection_wrapper(particle_old, particle_new)
        test.assert_almost_equal(intersection.x1_py+self.xmin, 2.0)
        test.assert_almost_equal(intersection.y1_py+self.ymin, 1.0)
        test.assert_almost_equal(intersection.x2_py+self.xmin, 1.0)
        test.assert_almost_equal(intersection.y2_py+self.ymin, 1.0)
        test.assert_almost_equal(intersection.xi_py+self.xmin, 1.5)
        test.assert_almost_equal(intersection.yi_py+self.ymin, 1.0)

    def test_get_boundary_intersection_when_a_particle_has_moved_to_an_external_elements_edge(self):
        particle_old = ParticleSmartPtr(x1=1.5-self.xmin, x2=1.1-self.ymin, host_elements={'fvcom': 1})
        particle_new = ParticleSmartPtr(x1=1.5-self.xmin, x2=1.0-self.ymin, host_elements={'fvcom': 1})
        intersection = self.data_reader.get_boundary_intersection_wrapper(particle_old, particle_new)
        test.assert_almost_equal(intersection.x1_py+self.xmin, 2.0)
        test.assert_almost_equal(intersection.y1_py+self.ymin, 1.0)
        test.assert_almost_equal(intersection.x2_py+self.xmin, 1.0)
        test.assert_almost_equal(intersection.y2_py+self.ymin, 1.0)
        test.assert_almost_equal(intersection.xi_py+self.xmin, 1.5)
        test.assert_almost_equal(intersection.yi_py+self.ymin, 1.0)

    def test_set_default_location(self):
        particle = ParticleSmartPtr(x1=1.5-self.xmin, x2=1.0-self.ymin, host_elements={'fvcom': 1})
        self.data_reader.set_default_location_wrapper(particle)
        test.assert_almost_equal(particle.x1 + self.xmin, 1.66666666667)
        test.assert_almost_equal(particle.x2 + self.ymin, 1.33333333333)
        test.assert_array_almost_equal(particle.get_phi('fvcom'), [0.3333333333, 0.3333333333, 0.3333333333])


    def test_set_local_coordinates_when_a_particle_is_on_an_external_elements_side(self):
        particle = ParticleSmartPtr(x1=1.5-self.xmin, x2=1.0-self.ymin, host_elements={'fvcom': 1})
        self.data_reader.set_local_coordinates_wrapper(particle)
        phi_min = np.min(np.array(particle.get_phi('fvcom'), dtype=float))
        test.assert_equal(np.abs(phi_min), 0.0)

    def test_get_zmin(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        host_elements={'fvcom': 0}

        time = 0.0
        
        particle = ParticleSmartPtr(x1=x1, x2=x2, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        bathy = self.data_reader.get_zmin_wrapper(time, particle)
        test.assert_almost_equal(bathy, -11.0)

    def test_get_zmax(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        host_elements={'fvcom': 0}
        
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        zeta = self.data_reader.get_zmax_wrapper(time, particle)
        test.assert_almost_equal(zeta, 1.0)
        
        time = 1800.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        zeta = self.data_reader.get_zmax_wrapper(time, particle)
        test.assert_almost_equal(zeta, 1.5)

    def test_set_vertical_grid_vars_for_a_particle_on_the_sea_surface(self):
        time = 0.0
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        x3 = 1.0
        host_elements={'fvcom': 0}

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        
        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 0)
        test.assert_equal(particle.in_vertical_boundary_layer, True)
        test.assert_almost_equal(particle.omega_interfaces, 1.0)

    def test_set_vertical_grid_vars_for_a_particle_on_the_sea_floor(self):
        time = 0.0
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        x3 = -11.0
        host_elements={'fvcom': 0}

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)        
        test.assert_equal(particle.k_layer, 2)
        test.assert_equal(particle.in_vertical_boundary_layer, True)
        test.assert_almost_equal(particle.omega_interfaces, 0.0)

    def test_set_vertical_grid_vars_for_a_particle_in_the_surface_boundary_layer(self):
        time = 0.0
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        x3 = 0.4 # this is 25% of the way between the top and bottom sigma levels
        host_elements={'fvcom': 0}

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 0)
        test.assert_equal(particle.in_vertical_boundary_layer, True)
        test.assert_almost_equal(particle.omega_interfaces, 0.75)

    def test_set_vertical_grid_vars_for_a_particle_in_the_bottom_boundary_layer(self):
        time = 0.0
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        x3 = -10.4
        host_elements={'fvcom': 0}

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 2)
        test.assert_equal(particle.in_vertical_boundary_layer, True)
        test.assert_almost_equal(particle.omega_interfaces, 0.25)

    def test_set_vertical_grid_vars_for_a_particle_in_the_middle_of_the_water_column(self):
        time = 0.0
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        x3 = -2.6
        host_elements={'fvcom': 0}

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 1)
        test.assert_equal(particle.in_vertical_boundary_layer, False)
        test.assert_equal(particle.k_upper_layer, 0)
        test.assert_equal(particle.k_lower_layer, 1)
        test.assert_almost_equal(particle.omega_interfaces, 0.83333333333)
        test.assert_almost_equal(particle.omega_layers, 0.5)

    def test_get_velocity_in_surface_layer(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        host_elements={'fvcom': 0}

        # Test #1
        x3 = 1.0
        time = 0.0

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)

        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [2.0, 2.0, 2.0])

        # Test #2
        x3 = 1.5
        time = 1800.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [3.0, 3.0, 3.0])

        # Test #3
        x3 = -0.2
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [2.0, 2.0, 2.0])

    def test_get_velocity_in_middle_layer(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        host_elements={'fvcom': 0}

        # Test #1
        x3 = -2.6
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [1.5, 1.5, 1.5])

        # Test #2
        x3 = -2.25
        time = 1800.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [2.25, 2.25, 2.25])

    def test_get_velocity_in_bottom_layer(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        host_elements={'fvcom': 0}

        # Test #1
        x3 = -10.999
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [0.0, 0.0, 0.0])

        # Test #2
        x3 = -10.999
        time = 1800.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [0.0, 0.0, 0.0])

        # Test #3
        x3 = -9.8
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [0.0, 0.0, 0.0])

    def test_get_thetao(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        host_elements={'fvcom': 0}

        x3 = -0.1
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        thetao = self.data_reader.get_environmental_variable_wrapper('thetao', time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(thetao,  0.01)

    def test_get_so(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        host_elements={'fvcom': 0}

        x3 = -0.1
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        so = self.data_reader.get_environmental_variable_wrapper('so', time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(so,  0.01)

    def test_get_vertical_eddy_diffusivity(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        host_elements={'fvcom': 0}

        x3 = -0.2
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        diffusivity = self.data_reader.get_vertical_eddy_diffusivity_wrapper(time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(diffusivity,  0.005)

    def test_get_vertical_eddy_diffusivity_derivative(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        host_elements={'fvcom': 0}

        x3 = -0.2
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        diffusivity_gradient = self.data_reader.get_vertical_eddy_diffusivity_derivative_wrapper(time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(diffusivity_gradient, -0.0026042)

    def test_get_horizontal_eddy_viscosity(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        host_elements={'fvcom': 0}

        x3 = -0.1
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        viscosity = self.data_reader.get_horizontal_eddy_viscosity_wrapper(time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(viscosity,  0.01)

    def test_get_horizontal_eddy_viscosity_derivative(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        host_elements={'fvcom': 0}

        x3 = -0.1
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        Ah_prime = np.empty(2, dtype=DTYPE_FLOAT)
        self.data_reader.get_horizontal_eddy_viscosity_derivative_wrapper(time, particle, Ah_prime)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(Ah_prime, [0.0, 0.0])

    def test_element_is_wet(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        host_elements={'fvcom': 0}
        time = 0.0

        particle = ParticleSmartPtr(x1=x1, x2=x2, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        status = self.data_reader.is_wet_wrapper(time, particle)
        test.assert_equal(status, 1)


class FVCOMReflectingHorizBoundaryCondition_test(TestCase):

    def setUp(self):
        # Create config
        config = configparser.SafeConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')
        config.add_section("SIMULATION")
        config.set('SIMULATION', 'time_direction', 'forward')
        config.add_section("OCEAN_CIRCULATION_MODEL")
        config.set('OCEAN_CIRCULATION_MODEL', 'coordinate_system', 'cartesian')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_Kh', 'True')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_Ah', 'True')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_is_wet', 'True')

        # Create mediator
        mediator = MockFVCOMMediator()
        
        # Create data reader
        self.data_reader = FVCOMDataReader(config, mediator)
        
        # Read in data
        datetime_start = datetime.datetime(2000,1,1) # Arbitrary start time, ignored by mock mediator
        datetime_end = datetime.datetime(2000,1,1) # Arbitrary end time, ignored by mock mediator
        self.data_reader.setup_data_access(datetime_start, datetime_end)

        # Grid offsets
        self.xmin = self.data_reader.get_xmin()
        self.ymin = self.data_reader.get_ymin()

        # Boundary condition calculator
        self.horiz_boundary_condition_calculator = RefHorizBoundaryConditionCalculator()

    def tearDown(self):
        del(self.data_reader)

    def test_reflect_particle_on_a_normal_trajectory(self):
        particle_old = ParticleSmartPtr(x1=1.6666666667 - self.xmin, x2=1.2 - self.ymin, host_elements={'fvcom': 1})
        particle_new = ParticleSmartPtr(x1=1.6666666667 - self.xmin, x2=0.9 - self.ymin, host_elements={'fvcom': 1})
        flag = self.horiz_boundary_condition_calculator.apply_wrapper(self.data_reader, particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.x1 + self.xmin, 1.6666666667)
        test.assert_equal(particle_new.x2 + self.ymin, 1.1)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 1)

    def test_reflect_particle_on_an_angled_trajectory(self):
        particle_old = ParticleSmartPtr(x1=1.4 - self.xmin, x2=1.1 - self.ymin, host_elements={'fvcom': 1})
        particle_new = ParticleSmartPtr(x1=1.6 - self.xmin, x2=0.9 - self.ymin, host_elements={'fvcom': 1})
        flag = self.horiz_boundary_condition_calculator.apply_wrapper(self.data_reader, particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.x1 + self.xmin, 1.6)
        test.assert_equal(particle_new.x2 + self.ymin, 1.1)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 1)

    def test_reflect_particle_that_sits_on_the_boundary(self):
        particle_old = ParticleSmartPtr(x1=1.5 - self.xmin, x2=1.0 - self.ymin, host_elements={'fvcom': 1})
        particle_new = ParticleSmartPtr(x1=1.6 - self.xmin, x2=0.9 - self.ymin, host_elements={'fvcom': 1})
        flag = self.horiz_boundary_condition_calculator.apply_wrapper(self.data_reader, particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.x1 + self.xmin, 1.6)
        test.assert_equal(particle_new.x2 + self.ymin, 1.1)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 1)

    def test_reflect_particle_that_undergoes_a_double_reflection_while_on_a_southeast_trajectory(self):
        particle_old = ParticleSmartPtr(x1=1.8 - self.xmin, x2=1.1 - self.ymin, host_elements={'fvcom': 1})
        particle_new = ParticleSmartPtr(x1=2.1 - self.xmin, x2=0.8 - self.ymin, host_elements={'fvcom': 1})
        flag = self.horiz_boundary_condition_calculator.apply_wrapper(self.data_reader, particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.x1 + self.xmin, 1.9)
        test.assert_equal(particle_new.x2 + self.ymin, 1.2)
        test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 1)


class FVCOMReflectingVertBoundaryCondition_test(TestCase):

    def setUp(self):
        # Create config
        config = configparser.SafeConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')
        config.add_section("SIMULATION")
        config.set('SIMULATION', 'time_direction', 'forward')
        config.add_section("OCEAN_CIRCULATION_MODEL")
        config.set('OCEAN_CIRCULATION_MODEL', 'coordinate_system', 'cartesian')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_Kh', 'True')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_Ah', 'True')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_is_wet', 'True')

        # Create mediator
        mediator = MockFVCOMMediator()
        
        # Create data reader
        self.data_reader = FVCOMDataReader(config, mediator)
        
        # Read in data
        datetime_start = datetime.datetime(2000,1,1) # Arbitrary start time, ignored by mock mediator
        datetime_end = datetime.datetime(2000,1,1) # Arbitrary end time, ignored by mock mediator
        self.data_reader.setup_data_access(datetime_start, datetime_end)

        # Grid offsets
        self.xmin = self.data_reader.get_xmin()
        self.ymin = self.data_reader.get_ymin()

        # Boundary condition calculator
        self.vert_boundary_condition_calculator = RefVertBoundaryConditionCalculator()

    def tearDown(self):
        del(self.data_reader)

    def test_apply_reflecting_boundary_condition_for_a_particle_that_has_pierced_the_free_surface(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        x3 = 1.0 + 0.1
        host_elements={'fvcom': 0}

        time = 0.0
        
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.vert_boundary_condition_calculator.apply_wrapper(self.data_reader, time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(particle.x3, 0.9)

    def test_apply_reflecting_boundary_condition_for_a_particle_that_has_just_pierced_the_free_surface(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        x3 = 1.0 + 1.e-14
        host_elements={'fvcom': 0}

        time = 0.0
        
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.vert_boundary_condition_calculator.apply_wrapper(self.data_reader, time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(particle.x3, 1.0)

    def test_apply_reflecting_boundary_condition_for_a_particle_that_has_pierced_the_sea_floor(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        x3 = -11.0 - 0.1
        host_elements={'fvcom': 0}

        time = 0.0
        
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.vert_boundary_condition_calculator.apply_wrapper(self.data_reader, time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(particle.x3, -10.9)

    def test_apply_reflecting_boundary_condition_for_a_particle_that_has_just_pierced_the_free_surface(self):
        x1 = 1.3333333333-self.xmin
        x2 = 1.6666666667-self.ymin
        x3 = -11.0 - 1.e-14
        host_elements={'fvcom': 0}

        time = 0.0
        
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.vert_boundary_condition_calculator.apply_wrapper(self.data_reader, time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(particle.x3, -11.0)

