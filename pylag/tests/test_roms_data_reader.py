"""
TODO - Fix unit tests for geographic coordinate case
"""

from unittest import TestCase
import numpy.testing as test
import numpy as np
import datetime
from scipy.spatial import Delaunay

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

# PyLag data types
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT

from pylag.roms_data_reader import ROMSDataReader
from pylag.particle_cpp_wrapper import ParticleSmartPtr

from pylag.mediator import Mediator
from pylag import grid_metrics as gm


class MockROMSMediator(Mediator):
    """ Test mediator for ROMS gridded data
    """
    def __init__(self):
        # Dictionaries for grid and time dependent variables
        self._dim_vars = {}
        self._grid_vars = {}
        self._time_dep_var_dimensions = {}
        self._time_dep_vars_last = {}
        self._time_dep_vars_next = {}

        # Basic grid (2 x 2 x 3) in rho
        latitude_rho = np.array([12.0, 14.0], dtype=DTYPE_FLOAT)
        longitude_rho = np.array([2., 4.], dtype=DTYPE_FLOAT)

        # Basic grid (2 x 2 x 3) in u
        latitude_u = np.array([12.0, 14.0], dtype=DTYPE_FLOAT)
        longitude_u = np.array([1., 3., 5.], dtype=DTYPE_FLOAT)

        latitude_v = np.array([11.0, 13.0, 15.], dtype=DTYPE_FLOAT)
        longitude_v = np.array([2., 4.], dtype=DTYPE_FLOAT)

        # Save horizontal grid vars for each grid
        for grid_name, lon, lat in zip(['grid_rho', 'grid_u', 'grid_v'],
                                       [longitude_rho, longitude_u, longitude_v],
                                       [latitude_rho, latitude_u, latitude_v]):

            # Dimension sizes
            n_longitude = len(lon)
            n_latitude = len(lat)

            # Form the unstructured grid
            lon2d, lat2d = np.meshgrid(lon[:], lat[:], indexing='ij')
            points = np.array([lon2d.flatten(order='C'), lat2d.flatten(order='C')]).T

            # Save lon and lat points at nodes
            lon_nodes = points[:, 0]
            lat_nodes = points[:, 1]
            n_nodes = points.shape[0]

            # Create the Triangulation
            tri = Delaunay(points)

            # Save simplices
            #   - Flip to reverse ordering, as expected by PyLag
            nv = np.asarray(np.flip(tri.simplices.copy(), axis=1), dtype=DTYPE_INT)
            n_elements = nv.shape[0]

            # Save neighbours
            #   - Transpose to give it the dimension ordering expected by PyLag
            #   - Sort to ensure match with nv
            nbe = np.asarray(tri.neighbors, dtype=DTYPE_INT)
            gm.sort_adjacency_array(nv, nbe)

            # Save lon and lat points at element centres
            lon_elements = np.empty(n_elements, dtype=DTYPE_FLOAT)
            lat_elements = np.empty(n_elements, dtype=DTYPE_FLOAT)
            for i, element in enumerate(range(n_elements)):
                lon_elements[i] = lon_nodes[(nv[element, :])].mean()
                lat_elements[i] = lat_nodes[(nv[element, :])].mean()

            # Transpose arrays
            nv = nv.T
            nbe = nbe.T

            # Flag open boundaries with -2 flag
            nbe[np.asarray(nbe == -1).nonzero()] = -2

            self._dim_vars['longitude_{}'.format(grid_name)] = n_longitude
            self._dim_vars['latitude_{}'.format(grid_name)] = n_latitude
            self._dim_vars['node_{}'.format(grid_name)] = n_nodes
            self._dim_vars['element_{}'.format(grid_name)] = n_elements

            self._grid_vars['longitude_{}'.format(grid_name)] = lon_nodes
            self._grid_vars['latitude_{}'.format(grid_name)] = lat_nodes
            self._grid_vars['longitude_c_{}'.format(grid_name)] = lon_elements
            self._grid_vars['latitude_c_{}'.format(grid_name)] = lat_elements

            self._grid_vars['nv_{}'.format(grid_name)] = nv
            self._grid_vars['nbe_{}'.format(grid_name)] = nbe

            self._grid_vars['mask_{}'.format(grid_name)] = np.zeros_like(lon_nodes)
            self._grid_vars['mask_c_{}'.format(grid_name)] = np.zeros_like(lon_elements)
            self._grid_vars['mask_n_{}'.format(grid_name)] = np.zeros_like(lon_elements)

        # Bathymetry at rho points [lat, lon]
        h = np.array([[20., 20.], [10., 10.]])
        h = np.moveaxis(h, 1, 0)  # Move to [lon, lat]
        h = h.reshape(np.prod(h.shape), order='C')
        self._grid_vars['h'] = h

        # Depth grid
        s_w_dimensions = ('s_w')
        s_rho_dimensions = ('s_rho')
        cs_w_dimensions = ('s_w')
        cs_r_dimensions = ('s_rho')
        s_w = np.array([-1., -0.5, 0.], dtype=DTYPE_FLOAT)
        cs_w = np.array([-1., -0.5, 0.], dtype=DTYPE_FLOAT)
        s_rho = np.array([-0.75, -0.25], dtype=DTYPE_FLOAT)
        cs_r = np.array([-0.75, -0.25], dtype=DTYPE_FLOAT)
        hc = np.array([20.], dtype=DTYPE_FLOAT)
        vtransform = np.array([2.], dtype=DTYPE_FLOAT)

        # Depth vars
        self._dim_vars['s_w'] = len(s_w)
        self._dim_vars['s_rho'] = len(s_rho)
        self._grid_vars['hc'] = hc
        self._grid_vars['s_w'] = s_w
        self._grid_vars['cs_w'] = cs_w
        self._grid_vars['s_rho'] = s_rho
        self._grid_vars['cs_r'] = cs_r
        self._grid_vars['vtransform'] = vtransform

        # Grid var dimensions
        self._grid_var_dimensions = {'s_w': ('s_w'),
                                     'cs_w': ('s_w'),
                                     's_rho': ('s_rho'),
                                     'cs_r': ('s_rho')}

        # Zeta at rho points [lat, lon]
        zos_dimensions = ('time', 'latitude_grid_rho', 'longitude_grid_rho')
        zos = np.array([[1., 1.], [1., 1.]], dtype=DTYPE_FLOAT)

        # u at u points
        u_dimensions = ('time', 's_rho', 'latitude_grid_u', 'longitude_grid_u')
        u = np.array([[[0, 1, 0], [0, 1, 0]],
                      [[0, 1, 0], [0, 1, 0]]], dtype=DTYPE_FLOAT)

        # v at v points
        v_dimensions = ('time', 's_rho', 'latitude_grid_v', 'longitude_grid_v')
        v = np.array([[[0, 0], [2, 2], [0, 0]],
                      [[0, 0], [2, 2], [0, 0]]], dtype=DTYPE_FLOAT)

        # w at w points
        w_dimensions = ('time', 's_w', 'latitude_grid_rho', 'longitude_grid_rho')
        w = np.array([[[3, 3], [3, 3]],
                      [[3, 3], [3, 3]],
                      [[3, 3], [3, 3]]], dtype=DTYPE_FLOAT)

        # ts at rho points
        ts_dimensions = ('time', 's_rho', 'latitude_grid_rho', 'longitude_grid_rho')
        ts = np.array([[[4, 4], [4, 4]],
                      [[4, 4], [4, 4]]], dtype=DTYPE_FLOAT)

        # kh at w points
        kh_dimensions = ('time', 's_w', 'latitude_grid_rho', 'longitude_grid_rho')
        kh = np.array([[[5, 5], [5, 5]],
                      [[5, 5], [5, 5]],
                      [[5, 5], [5, 5]]], dtype=DTYPE_FLOAT)

        # Ah at w points
        ah_dimensions = ('time', 's_rho', 'latitude_grid_rho', 'longitude_grid_rho')
        ah = np.array([[[6, 6], [6, 6]],
                      [[6, 6], [6, 6]]], dtype=DTYPE_FLOAT)

        # Set dimensions
        self._time_dep_var_dimensions = {'s_w': s_w_dimensions, 's_rho': s_rho_dimensions,
                                         'cs_w': cs_w_dimensions, 'cs_r': cs_r_dimensions,
                                         'zos': zos_dimensions, 'uo': u_dimensions, 'vo': v_dimensions,
                                         'wo': w_dimensions, 'thetao': ts_dimensions, 'so': ts_dimensions,
                                         'kh': kh_dimensions, 'ah': ah_dimensions}

        # Store in dictionaries
        self._time_dep_vars_last = {'zos': zos, 'uo': u, 'vo': v, 'wo': w, 'thetao': ts,
                                    'so': ts, 'kh': kh, 'ah': ah}
        self._time_dep_vars_next = {'zos': zos, 'uo': u, 'vo': v, 'wo': w, 'thetao': ts,
                                    'so': ts, 'kh': kh, 'ah': ah}

        # Time in seconds. ie two time pts, 1 hour apart
        self._t_last = 0.0
        self._t_next = 3600.0

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

    def get_grid_variable(self, var_name, var_dims, var_type):
        return self._grid_vars[var_name][:].astype(var_type)

    def get_grid_variable_dimensions(self, var_name):
        return self._grid_var_dimensions[var_name]

    def get_variable_dimensions(self, var_name):
        return self._time_dep_var_dimensions[var_name]

    def get_variable_shape(self, var_name):
        return self._time_dep_vars_last[var_name].shape

    def get_time_dependent_variable_at_last_time_index(self, var_name, var_dims, var_type):
        var = self._time_dep_vars_last[var_name][:].astype(var_type)

        if np.ma.isMaskedArray(var):
            return var.data

        return var

    def get_time_dependent_variable_at_next_time_index(self, var_name, var_dims, var_type):
        var = self._time_dep_vars_next[var_name][:].astype(var_type)

        if np.ma.isMaskedArray(var):
            return var.data

        return var

    def get_mask_at_last_time_index(self, var_name, var_dims):
        var = self._time_dep_vars_last[var_name][:]

        if np.ma.isMaskedArray(var):
            return var.mask.astype(DTYPE_INT)

        raise RuntimeError('Variable {} is not a masked array.'.format(var_name))

    def get_mask_at_next_time_index(self, var_name, var_dims):
        var = self._time_dep_vars_next[var_name][:]

        if np.ma.isMaskedArray(var):
            return var.mask.astype(DTYPE_INT)

        raise RuntimeError('Variable {} is not a masked array.'.format(var_name))


class ROMSReader_test(TestCase):

    def setUp(self):
        self.deg_to_radians = np.radians(1)

        # Create config
        config = configparser.ConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')
        config.add_section("SIMULATION")
        config.set('SIMULATION', 'time_direction', 'forward')
        config.add_section("OCEAN_CIRCULATION_MODEL")
        config.set('OCEAN_CIRCULATION_MODEL', 'grid_type', 'rectilinear')
        config.set('OCEAN_CIRCULATION_MODEL', 'time_dim_name', 'time')
        config.set('OCEAN_CIRCULATION_MODEL', 'depth_dim_name_grid_rho', 's_rho')
        config.set('OCEAN_CIRCULATION_MODEL', 'depth_dim_name_grid_w', 's_w')
        config.set('OCEAN_CIRCULATION_MODEL', 'latitude_dim_name_grid_rho', 'latitude_grid_rho')
        config.set('OCEAN_CIRCULATION_MODEL', 'longitude_dim_name_grid_rho', 'longitude_grid_rho')
        config.set('OCEAN_CIRCULATION_MODEL', 'latitude_dim_name_grid_u', 'latitude_grid_u')
        config.set('OCEAN_CIRCULATION_MODEL', 'longitude_dim_name_grid_u', 'longitude_grid_u')
        config.set('OCEAN_CIRCULATION_MODEL', 'latitude_dim_name_grid_v', 'latitude_grid_v')
        config.set('OCEAN_CIRCULATION_MODEL', 'longitude_dim_name_grid_v', 'longitude_grid_v')
        config.set('OCEAN_CIRCULATION_MODEL', 'time_var_name', 'time')
        config.set('OCEAN_CIRCULATION_MODEL', 'uo_var_name', 'uo')
        config.set('OCEAN_CIRCULATION_MODEL', 'vo_var_name', 'vo')
        config.set('OCEAN_CIRCULATION_MODEL', 'wo_var_name', 'wo')
        config.set('OCEAN_CIRCULATION_MODEL', 'zos_var_name', 'zos')
        config.set('OCEAN_CIRCULATION_MODEL', 'thetao_var_name', 'thetao')
        config.set('OCEAN_CIRCULATION_MODEL', 'so_var_name', 'so')
        config.set('OCEAN_CIRCULATION_MODEL', 'Kh_var_name', 'kh')
        config.set('OCEAN_CIRCULATION_MODEL', 'Ah_var_name', 'ah')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_is_wet', 'False')
        config.set('OCEAN_CIRCULATION_MODEL', 'coordinate_system', 'geographic')
        config.add_section("OUTPUT")
        config.set('OUTPUT', 'environmental_variables', 'thetao, so')

        # Create mediator
        mediator = MockROMSMediator()
        
        # Create data reader
        self.data_reader = ROMSDataReader(config, mediator)
        
        # Read in data
        datetime_start = datetime.datetime(2000, 1, 1)  # Arbitrary start time, ignored by mock mediator
        datetime_end = datetime.datetime(2000, 1, 1)  # Arbitrary end time, ignored by mock mediator
        self.data_reader.setup_data_access(datetime_start, datetime_end)

        # Grid offsets
        self.xmin = self.data_reader.get_xmin()
        self.ymin = self.data_reader.get_ymin()

    def tearDown(self):
        del(self.data_reader)

    def test_find_host_using_global_search(self):
        particle = ParticleSmartPtr(x1=self.deg_to_radians * 2.25-self.xmin, x2=self.deg_to_radians * 13.75-self.ymin)
        flag = self.data_reader.find_host_using_global_search_wrapper(particle)
        test.assert_equal(flag, 0)
        test.assert_equal(particle.get_host_horizontal_elem('grid_rho'), 0)
        test.assert_equal(particle.get_host_horizontal_elem('grid_u'), 1)
        test.assert_equal(particle.get_host_horizontal_elem('grid_v'), 3)

    def test_find_host_when_particle_is_in_the_domain(self):
        particle_old = ParticleSmartPtr(x1=self.deg_to_radians * 2.25-self.xmin, x2=self.deg_to_radians * 13.75-self.ymin,
                                        host_elements={'grid_rho': 0, 'grid_u': 1, 'grid_v': 3})
        particle_new = ParticleSmartPtr(x1=self.deg_to_radians * 3.75-self.xmin, x2=self.deg_to_radians * 12.25-self.ymin,
                                        host_elements={'grid_rho': -999, 'grid_u': -999, 'grid_v': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.get_host_horizontal_elem('grid_rho'), 1)
        test.assert_equal(particle_new.get_host_horizontal_elem('grid_u'), 3)
        test.assert_equal(particle_new.get_host_horizontal_elem('grid_v'), 1)

    # def test_get_zmin(self):
    #     particle = ParticleSmartPtr(x1=2.666666667-self.xmin, x2=13.333333333-self.ymin,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 3})
    #     time = 0.0
    #
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     bathy = self.data_reader.get_zmin_wrapper(time, particle)
    #     test.assert_almost_equal(bathy, -13.333333333333)
    #
    # def test_get_zmax(self):
    #     particle = ParticleSmartPtr(x1=2.666666667-self.xmin, x2=13.333333333-self.ymin,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 3})
    #
    #     time = 0.0
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     zeta = self.data_reader.get_zmax_wrapper(time, particle)
    #     test.assert_almost_equal(zeta, 1.0)
    #
    #     time = 1800.0
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     zeta = self.data_reader.get_zmax_wrapper(time, particle)
    #     test.assert_almost_equal(zeta, 1.0)
    #
    # def test_set_vertical_grid_vars_for_a_particle_on_the_sea_surface(self):
    #     particle = ParticleSmartPtr(x1=2.666666667-self.xmin, x2=13.333333333-self.ymin, x3=1.0,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 3})
    #
    #     time = 0.0
    #
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #
    #     test.assert_equal(flag, 0)
    #     test.assert_equal(particle.k_layer, 0)
    #     test.assert_equal(particle.in_vertical_boundary_layer, True)
    #     test.assert_almost_equal(particle.omega_interfaces, 1.0)
    #
    # def test_set_vertical_grid_vars_for_a_particle_on_the_sea_floor(self):
    #     particle = ParticleSmartPtr(x1=2.666666667-self.xmin, x2=13.333333333-self.ymin, x3=-13.333333333,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 3})
    #
    #     time = 0.0
    #
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #
    #     test.assert_equal(flag, 0)
    #     test.assert_equal(particle.k_layer, 1)
    #     test.assert_equal(particle.in_vertical_boundary_layer, True)
    #     test.assert_almost_equal(particle.omega_interfaces, 0.0)
    #
    # def test_set_vertical_grid_vars_for_a_particle_in_the_surface_layer(self):
    #     particle = ParticleSmartPtr(x1=2-self.xmin, x2=12-self.ymin, x3=-6.875,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 0})
    #
    #     time = 0.0
    #
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #
    #     test.assert_equal(flag, 0)
    #     test.assert_equal(particle.k_layer, 0)
    #     test.assert_equal(particle.k_upper_layer, 0)
    #     test.assert_equal(particle.k_lower_layer, 1)
    #     test.assert_equal(particle.in_vertical_boundary_layer, False)
    #     test.assert_almost_equal(particle.omega_interfaces, 0.25)
    #     test.assert_almost_equal(particle.omega_layers, 0.75)
    #
    # def test_get_velocity_in_surface_layer(self):
    #     particle = ParticleSmartPtr(x1=3.-self.xmin, x2=13.-self.ymin, x3=1.0,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 0})
    #
    #     time = 0.0
    #
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #     test.assert_equal(flag, 0)
    #
    #     vel = np.empty(3, dtype=DTYPE_FLOAT)
    #     self.data_reader.get_velocity_wrapper(time, particle, vel)
    #     test.assert_array_almost_equal(vel, [1., 2., 3.])
    #
    # def test_get_thetao(self):
    #     particle = ParticleSmartPtr(x1=3.-self.xmin, x2=13.-self.ymin, x3=1.0,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 0})
    #
    #     time = 0.0
    #
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #     test.assert_equal(flag, 0)
    #
    #     thetao = self.data_reader.get_environmental_variable_wrapper('thetao', time, particle)
    #     test.assert_almost_equal(thetao,  4.0)
    #
    # def test_get_so(self):
    #     particle = ParticleSmartPtr(x1=3.-self.xmin, x2=13.-self.ymin, x3=1.0,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 0})
    #
    #     time = 0.0
    #
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #     test.assert_equal(flag, 0)
    #
    #     so = self.data_reader.get_environmental_variable_wrapper('so', time, particle)
    #     test.assert_almost_equal(so,  4.0)
    #
    # def test_get_vertical_eddy_diffusivity(self):
    #     particle = ParticleSmartPtr(x1=3.-self.xmin, x2=13.-self.ymin, x3=1.0,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 0})
    #
    #     time = 0.0
    #
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #     test.assert_equal(flag, 0)
    #
    #     diffusivity = self.data_reader.get_vertical_eddy_diffusivity_wrapper(time, particle)
    #     test.assert_almost_equal(diffusivity,  5.0)
    #
    # def test_get_vertical_eddy_diffusivity_derivative(self):
    #     particle = ParticleSmartPtr(x1=3. - self.xmin, x2=13. - self.ymin, x3=1.0,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 0})
    #
    #     time = 0.0
    #
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #     test.assert_equal(flag, 0)
    #
    #     diffusivity_derivative = self.data_reader.get_vertical_eddy_diffusivity_derivative_wrapper(time, particle)
    #     test.assert_almost_equal(diffusivity_derivative, 0.0)
    #
    # def test_get_horizontal_eddy_viscosity(self):
    #     particle = ParticleSmartPtr(x1=3.-self.xmin, x2=13.-self.ymin, x3=1.0,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 0})
    #
    #     time = 0.0
    #
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #     test.assert_equal(flag, 0)
    #
    #     diffusivity = self.data_reader.get_horizontal_eddy_viscosity_wrapper(time, particle)
    #     test.assert_almost_equal(diffusivity,  6.0)
    #
    # def test_get_horizontal_eddy_viscosity_derivative(self):
    #     particle = ParticleSmartPtr(x1=3. - self.xmin, x2=13. - self.ymin, x3=1.0,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 0})
    #
    #     time = 0.0
    #
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #     test.assert_equal(flag, 0)
    #
    #     Ah_prime = np.empty(2, dtype=DTYPE_FLOAT)
    #     self.data_reader.get_horizontal_eddy_viscosity_derivative_wrapper(time, particle, Ah_prime)
    #     test.assert_equal(flag, 0)
    #     test.assert_array_almost_equal(Ah_prime, [0.0, 0.0])
    #
    # def test_element_is_wet(self):
    #     particle = ParticleSmartPtr(x1=3. - self.xmin, x2=13. - self.ymin, x3=1.0,
    #                                 host_elements={'grid_rho': 0, 'grid_u': 0, 'grid_v': 0})
    #
    #     time = 0.0
    #
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #     test.assert_equal(flag, 0)
    #     status = self.data_reader.is_wet_wrapper(time, particle)
    #     test.assert_equal(status, 1)

#    def test_element_is_dry(self):
#        pass



