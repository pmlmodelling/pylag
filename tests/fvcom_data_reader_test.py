from unittest import TestCase
import numpy.testing as test
import numpy as np
import datetime
from ConfigParser import SafeConfigParser

from pylag.fvcom_data_reader import FVCOMDataReader
from pylag import cwrappers

from pylag.mediator import Mediator

class MockFVCOMMediator(Mediator):
    """ Test mediator for FVCOM

    """
    def __init__(self):
        # Dimension variables
        n_elems = 4
        n_nodes = 6
        n_siglev = 4
        n_siglay = 3
        self._dim_vars = {'nele': n_elems, 'node': n_nodes,
                          'siglev': n_siglev, 'siglay': n_siglay}
                          
        # Grid variables - copied from a real FVCOM output file
        nv = np.array([[4,0,2,4],[3,1,4,5],[1,3,1,3]],dtype=int)
        nbe = np.array([[1,0,0,-1],[2,-1,-1,0],[3,-1,-1,-1]],dtype=int)
        x = np.array([370395.625, 367130.84375, 363094.219, 370100., 367030., 367652.656], dtype=float)
        y = np.array([5321986.5, 5322253., 5322582.5, 5325070., 5325870., 5328112.5], dtype=float)
        xc = np.array([368086.9, 369208.8, 365751.7, 368260.9], dtype=float)
        yc = np.array([5324398.5, 5323103., 5323568.5, 5326351.], dtype=float)
        siglay = np.array([[-.1,-.5,-.9],]*n_nodes, dtype=float).transpose()
        siglev = np.array([[0., -.2,-.8,-1.],]*n_nodes, dtype=float).transpose()
        h = np.array([10., 11., 10., 11., 11., 10.], dtype=float)
        a1u = np.array([[0.0001532937, 0, 0, 0.0001793663],
                        [0.0001951739, 0, 0, 0.0001998931],
                        [-0.0003354263, 0, 0, -1.715655e-05],
                        [-1.304125e-05, 0, 0, -0.0003621029]], dtype=float)
        a2u = np.array([[7.157444e-06, 0, 0, 0.0001338511],
                        [-0.0002355544, 0, 0, 0.0001577292],
                        [-8.94851e-05, 0, 0, -0.0004162626],
                        [0.0003178821, 0, 0, 0.0001246824]], dtype=float)
        self._grid_vars = {'nv': nv, 'nbe': nbe, 'x': x, 'y': y, 'xc': xc,
                           'yc': yc, 'siglay': siglay, 'siglev': siglev,
                           'h': h, 'a1u': a1u, 'a2u': a2u}
        
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

        # Wet cells
        wet_cells = np.array([[1,1,1,1],[1,1,1,1]],dtype=int)

        # Store in dictionaries
        self._time_dep_vars_last = {'zeta': zeta[0,:], 'u': u[0,:], 'v': v[0,:],
                                    'ww': ww[0,:], 'kh': kh[0,:],
                                    'viscofh': viscofh[0,:],
                                    'wet_cells': wet_cells[0,:]}
        self._time_dep_vars_next = {'zeta': zeta[1,:], 'u': u[1,:], 'v': v[1,:],
                                    'ww': ww[1,:], 'kh': kh[1,:],
                                    'viscofh': viscofh[1,:],
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
        config = SafeConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')
        config.add_section("OCEAN_CIRCULATION_MODEL")
        config.set('OCEAN_CIRCULATION_MODEL', 'has_Kh', 'True')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_Ah', 'True')

        # Create mediator
        mediator = MockFVCOMMediator()
        
        # Create data reader
        self.data_reader = FVCOMDataReader(config, mediator)
        
        # Read in data
        datetime_start = datetime.datetime(2000,1,1) # Arbitrary start time, ignored by mock mediator
        datetime_end = datetime.datetime(2000,1,1) # Arbitrary end time, ignored by mock mediator
        self.data_reader.setup_data_access(datetime_start, datetime_end)

    def tearDown(self):
        del(self.data_reader)

    def test_find_host_using_global_search(self):
        xpos = 368086.9375 # Centroid of element 0 (x coordinate)
        ypos = 5324397.5 # Centroid of element 0 (y coordinate)
        host = self.data_reader.find_host_using_global_search(xpos, ypos)
        test.assert_equal(host, 0)

    def test_find_host_using_global_search_when_a_particle_is_in_an_element_with_two_land_boundaries(self):
        xpos = 369208.8125 # Centroid of element 1 that has two land boundaries (x coordinate)
        ypos = 5323103.0 # Centroid of element 1 that has two land boundaries (y coordinate)
        host = self.data_reader.find_host_using_global_search(xpos, ypos)
        test.assert_equal(host, -1)

    def test_find_host_when_particle_is_in_the_domain(self):
        xpos_old = 368260.875 # Centroid of element 3 (x coordinate)
        ypos_old = 5326351.0 # Centroid of element 3 (y coordinate)
        xpos_new = 368086.9375 # Centroid of element 0 (x coordinate)
        ypos_new = 5324397.5 # Centroid of element 0 (y coordinate)
        last_host = 3
        flag, host = self.data_reader.find_host(xpos_old, ypos_old, xpos_new,
                ypos_new, last_host)
        test.assert_equal(flag, 0)
        test.assert_equal(host, 0)

    def test_find_host_when_particle_has_crossed_into_an_element_with_two_land_boundaries(self):
        xpos_old = 368086.9375 # Centroid of element 0 (x coordinate)
        ypos_old = 5324397.5 # Centroid of element 0 (y coordinate)
        xpos_new = 369208.8125 # Centroid of element 1 that has two land boundaries (x coordinate)
        ypos_new = 5323103.0 # Centroid of element 1 that has two land boundaries (y coordinate)
        last_host = 0 # Center element
        flag, host = self.data_reader.find_host(xpos_old, ypos_old, xpos_new,
                ypos_new, last_host)
        test.assert_equal(flag, -1)
        test.assert_equal(host, 0)

    def test_find_host_when_particle_has_crossed_a_land_boundary(self):
        xpos_old = 369208.8125 # Centroid of element 1 (x coordinate)
        ypos_old = 5323103.0 # Centroid of element 1 (y coordinate)
        xpos_new = 370267.0 # Point outside of the domain (x coordinate)
        ypos_new = 5324350.0 # Point outside of the domain (y coordinate)
        last_host = 1 # Center element
        flag, host = self.data_reader.find_host(xpos_old, ypos_old, xpos_new,
                ypos_new, last_host)
        test.assert_equal(flag, -1)
        test.assert_equal(host, 1)

    def test_find_host_when_particle_has_crossed_multiple_elements_to_an_element_with_two_land_boundaries(self):
        xpos_old = 369208.8125 # Centroid of element 1 (x coordinate)
        ypos_old = 5323103.0 # Centroid of element 1 (y coordinate)
        xpos_new = 365751.6875 # Centroid of element 2 (x coordinate)
        ypos_new = 5323568.5 # Centroid of element 2 (y coordinate)
        last_host = 1 # Center element
        flag, host = self.data_reader.find_host(xpos_old, ypos_old, xpos_new,
                ypos_new, last_host)
        test.assert_equal(flag, -1)
        test.assert_equal(host, 0)

    def test_get_boundary_intersection_x2x0(self):
        xpos_old = 369208.8125 # Centroid of element 1 (x coordinate)
        ypos_old = 5323103.0 # Centroid of element 1 (y coordinate)
        xpos_new = 370267.0 # Point outside element (x coordinate)
        ypos_new = 5324350.0 # Point outside element (y coordinate)
        last_host = 1
        x1, y1, x2, y2, xi, yi = self.data_reader.get_boundary_intersection(xpos_old, ypos_old, xpos_new, ypos_new, last_host)
        test.assert_almost_equal(x1, 370100.0)
        test.assert_almost_equal(y1, 5325070.0)
        test.assert_almost_equal(x2, 370395.625)
        test.assert_almost_equal(y2, 5321986.5)

    def test_get_boundary_intersection_x0x1(self):
        xpos_old = 369208.8125 # Centroid of element 1 (x coordinate)
        ypos_old = 5323103.0 # Centroid of element 1 (y coordinate)
        xpos_new = 368802.0 # Point outside element (x coordinate)
        ypos_new = 5321920.0 # Point outside element (y coordinate)
        last_host = 1
        x1, y1, x2, y2, xi, yi = self.data_reader.get_boundary_intersection(xpos_old, ypos_old, xpos_new, ypos_new, last_host)
        test.assert_almost_equal(x1, 370395.625)
        test.assert_almost_equal(y1, 5321986.5)
        test.assert_almost_equal(x2, 367130.84375)
        test.assert_almost_equal(y2, 5322253.0)

    def test_get_zmin(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0

        time = 0.0
        bathy = cwrappers.get_zmin(self.data_reader, time, xpos, ypos, host)
        test.assert_almost_equal(bathy, -11.0)

    def test_get_zmax(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0
        
        time = 0.0
        zeta = cwrappers.get_zmax(self.data_reader, time, xpos, ypos, host)
        test.assert_almost_equal(zeta, 1.0)
        
        time = 1800.0
        zeta = cwrappers.get_zmax(self.data_reader, time, xpos, ypos, host)
        test.assert_almost_equal(zeta, 1.5)

    def test_set_vertical_grid_vars_for_a_particle_on_the_sea_surface(self):
        time = 0.0
        xpos = 365751.7
        ypos = 5323568.0
        zpos = 1.0
        host = 0

        grid_vars = cwrappers.set_vertical_grid_vars(self.data_reader, time, xpos, ypos, zpos, host)
        
        test.assert_equal(grid_vars['k_layer'], 0)
        test.assert_equal(grid_vars['in_vertical_boundary_layer'], True)
        test.assert_almost_equal(grid_vars['omega_interfaces'], 1.0)

    def test_set_vertical_grid_vars_for_a_particle_on_the_sea_floor(self):
        time = 0.0
        xpos = 365751.7
        ypos = 5323568.0
        zpos = -11.0
        host = 0

        grid_vars = cwrappers.set_vertical_grid_vars(self.data_reader, time, xpos, ypos, zpos, host)
        
        test.assert_equal(grid_vars['k_layer'], 2)
        test.assert_equal(grid_vars['in_vertical_boundary_layer'], True)
        test.assert_almost_equal(grid_vars['omega_interfaces'], 0.0)


    def test_set_vertical_grid_vars_for_a_particle_in_the_surface_boundary_layer(self):
        time = 0.0
        xpos = 365751.7
        ypos = 5323568.0
        zpos = 0.4 # this is 25% of the way between the top and bottom sigma levels
        host = 0

        grid_vars = cwrappers.set_vertical_grid_vars(self.data_reader, time, xpos, ypos, zpos, host)
        
        test.assert_equal(grid_vars['k_layer'], 0)
        test.assert_equal(grid_vars['in_vertical_boundary_layer'], True)
        test.assert_almost_equal(grid_vars['omega_interfaces'], 0.75)

    def test_set_vertical_grid_vars_for_a_particle_in_the_bottom_boundary_layer(self):
        time = 0.0
        xpos = 365751.7
        ypos = 5323568.0
        zpos = -10.4
        host = 0

        grid_vars = cwrappers.set_vertical_grid_vars(self.data_reader, time, xpos, ypos, zpos, host)
        
        test.assert_equal(grid_vars['k_layer'], 2)
        test.assert_equal(grid_vars['in_vertical_boundary_layer'], True)
        test.assert_almost_equal(grid_vars['omega_interfaces'], 0.25)

    def test_set_vertical_grid_vars_for_a_particle_in_the_middle_of_the_water_column(self):
        time = 0.0
        xpos = 365751.7
        ypos = 5323568.0
        zpos = -2.6
        host = 0

        grid_vars = cwrappers.set_vertical_grid_vars(self.data_reader, time, xpos, ypos, zpos, host)
        
        test.assert_equal(grid_vars['k_layer'], 1)
        test.assert_equal(grid_vars['in_vertical_boundary_layer'], False)
        test.assert_equal(grid_vars['k_upper_layer'], 0)
        test.assert_equal(grid_vars['k_lower_layer'], 1)
        test.assert_almost_equal(grid_vars['omega_interfaces'], 0.83333333333)
        test.assert_almost_equal(grid_vars['omega_layers'], 0.5)

    def test_get_velocity_in_surface_layer(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0
        zlayer = 0

        zpos = 1.0
        time = 0.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host, zlayer)
        test.assert_array_almost_equal(vel, [2.0, 2.0, 2.0])

        zpos = 1.5
        time = 1800.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host, zlayer)
        test.assert_array_almost_equal(vel, [3.0, 3.0, 3.0])
        
        zpos = -0.2
        time = 0.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host, zlayer)
        test.assert_array_almost_equal(vel, [2.0, 2.0, 2.0])

    def test_get_velocity_in_middle_layer(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0
        zlayer = 1

        zpos = -2.6
        time = 0.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host, zlayer)
        test.assert_array_almost_equal(vel, [1.5, 1.5, 1.5])

        zpos = -2.25
        time = 1800.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host, zlayer)
        test.assert_array_almost_equal(vel, [2.25, 2.25, 2.25])

    def test_get_velocity_in_bottom_layer(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0
        zlayer = 2

        zpos = -10.999
        time = 0.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host, zlayer)
        test.assert_array_almost_equal(vel, [0.0, 0.0, 0.0])

        zpos = -10.999
        time = 1800.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host, zlayer)
        test.assert_array_almost_equal(vel, [0.0, 0.0, 0.0])
        
        zpos = -9.8
        time = 0.0
        vel = cwrappers.get_velocity(self.data_reader, time, xpos, ypos, zpos, host, zlayer)
        test.assert_array_almost_equal(vel, [0.0, 0.0, 0.0])

    def test_get_vertical_eddy_diffusivity(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0

        zpos = -0.2
        time = 0.0
        diffusivity = cwrappers.get_vertical_eddy_diffusivity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_almost_equal(diffusivity,  0.005)

    def test_get_vertical_eddy_diffusivity_derivative(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0

        zpos = -0.2
        time = 0.0

        diffusivity_gradient = cwrappers.get_vertical_eddy_diffusivity_derivative(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_almost_equal(diffusivity_gradient, -0.004166666666667)

    def test_get_horizontal_eddy_viscosity(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0

        zpos = -0.1
        time = 0.0
        diffusivity = cwrappers.get_horizontal_eddy_viscosity(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_almost_equal(diffusivity,  0.01)

    def test_get_horizontal_eddy_viscosity_derivative(self):
        xpos = 365751.7
        ypos = 5323568.0
        host = 0

        zpos = -0.1
        time = 0.0

        diffusivity_gradient = cwrappers.get_horizontal_eddy_viscosity_derivative(self.data_reader, time, xpos, ypos, zpos, host)
        test.assert_almost_equal(diffusivity_gradient, 0.0)

    def test_element_is_wet(self):
        host = 0
        time = 0.0

        status = self.data_reader.is_wet(time, host)
        test.assert_equal(status, 1)

