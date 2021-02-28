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

from pylag.arakawa_a_data_reader import ArakawaADataReader
from pylag.boundary_conditions import RefHorizGeographicBoundaryConditionCalculator as RefHorizBoundaryConditionCalculator
from pylag.particle_cpp_wrapper import ParticleSmartPtr

from pylag.mediator import Mediator
from pylag import grid_metrics as gm


class MockArakawaAMediator(Mediator):
    """ Test mediator for Arakawa-a gridded data

    We first define a structured 3x3x3 grid with depth, lat, and lon dimensions, in that
    order. From this we construct the unstructured grid using the approach adopted within
    create_arakawa_a_grid_metrics_file() in grid_metrics.py. This gives the unstructured
    grid variables (lat, lon, lat_c, lon_c, depth, h, land_sea_mask) to be readby PyLag.
    Next, we create the masked, time dependent variables (zeta, u, v, w). These are kept
    on their native structured grid - it is down to PyLag to correctly interpret and handle
    these, with accompanying unit tests put in place to ensure this is so.
    """
    def __init__(self):
        self.deg_to_radians = np.radians(1)

        # Basic grid (4 x 3 x 4).
        latitude = np.array([11., 12., 13., 14], dtype=DTYPE_FLOAT)
        longitude = np.array([1., 2., 3.], dtype=DTYPE_FLOAT)
        depth = np.array([0., 10., 20., 30.], dtype=DTYPE_FLOAT)

        # Save original grid dimensions
        n_latitude = latitude.shape[0]
        n_longitude = longitude.shape[0]
        n_depth = depth.shape[0]

        # Bathymetry [lat, lon]
        h = np.array([[25., 25., 25.], [10., 10., 10.], [999., 999., 999.], [999., 999., 999.]], dtype=DTYPE_FLOAT)
        h = np.moveaxis(h, 1, 0)  # Move to [lon, lat]
        h = h.reshape(np.prod(h.shape), order='C')

        # Mask [depth, lat, lon]. 1 is sea, 0 land. Note the last depth level is masked everywhere.
        mask = np.array([[[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
                         [[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
                         [[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=DTYPE_INT)

        # Switch the mask convention to that which PyLag anticipates. i.e. 1 is a masked point, 0 a non-masked point.
        mask = 1 - mask

        # Separately save the surface mask at nodes. This is taken as the land sea mask.
        land_sea_mask = mask[0, :, :]
        land_sea_mask_nodes = np.moveaxis(land_sea_mask, 0, 1)  # Move to [lon, lat]
        land_sea_mask_nodes = land_sea_mask_nodes.reshape(np.prod(land_sea_mask_nodes.shape), order='C')

        # Zeta (time = 0) [lat, lon]
        zeta_t0 = np.ma.masked_array([[1., 1., 1.], [0., 0., 0.], [999., 999., 999.], [999., 999., 999.]],
                                     mask=land_sea_mask, dtype=DTYPE_FLOAT)
        zeta_t1 = np.ma.copy(zeta_t0)

        # Mask one extra point (node 1) to help with testing wet dry status calls
        mask[:, 1, 0] = 1

        # u/v/w (time = 0) [depth, lat, lon]. Include mask.
        uvw_t0 = np.ma.masked_array([[[2., 2., 2.], [1., 1., 1.], [999., 999., 999.], [999., 999., 999.]],
                                     [[1., 1., 1.], [0., 0., 0.], [999., 999., 999.], [999., 999., 999.]],
                                     [[0., 0., 0.], [999., 999., 999.], [999., 999., 999.], [999., 999., 999.]],
                                     [[999., 999., 999.], [999., 999., 999.], [999., 999., 999.], [999., 999., 999.]]],
                                    mask=mask, dtype=DTYPE_FLOAT)
        uvw_t1 = np.ma.copy(uvw_t0)

        # t/s (time = 0) [depth, lat, lon]. Include mask.
        ts_t0 = np.ma.masked_array([[[2., 2., 2.], [1., 1., 1.], [999., 999., 999.], [999., 999., 999.]],
                                    [[1., 1., 1.], [0., 0., 0.], [999., 999., 999.], [999., 999., 999.]],
                                    [[0., 0., 0.], [999., 999., 999.], [999., 999., 999.], [999., 999., 999.]],
                                    [[999., 999., 999.], [999., 999., 999.], [999., 999., 999.], [999., 999., 999.]]],
                                   mask=mask, dtype=DTYPE_FLOAT)
        ts_t1 = np.ma.copy(ts_t0)

        # Trim latitudes
        trim_first_latitude = np.array([False])
        trim_last_latitude = np.array([False])

        # Form the unstructured grid
        lon2d, lat2d = np.meshgrid(longitude[:], latitude[:], indexing='ij')
        points = np.array([lon2d.flatten(order='C'), lat2d.flatten(order='C')]).T

        # Save lon and lat points at nodes
        lon_nodes = points[:, 0]
        lat_nodes = points[:, 1]
        n_nodes = points.shape[0]

        # Record the node permutation
        permutation = np.arange(n_nodes, dtype=DTYPE_INT)

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
        lon_elements, lat_elements = gm.compute_element_midpoints_in_geographic_coordinates(nv, lon_nodes, lat_nodes)

        # Generate the land-sea mask for elements
        land_sea_mask_elements = np.empty(n_elements, dtype=DTYPE_INT)
        gm.compute_land_sea_element_mask(nv, land_sea_mask_nodes, land_sea_mask_elements)

        # Transpose arrays
        nv = nv.T
        nbe = nbe.T

        # Flag open boundaries with -2 flag
        nbe[np.asarray(nbe == -1).nonzero()] = -2

        # Flag land boundaries with -1 flag
        #land_elements = np.asarray(land_sea_mask_elements == 1).nonzero()[0]
        #for element in land_elements:
        #    nbe[np.asarray(nbe == element).nonzero()] = -1

        # Add to grid dimensions and variables
        self._dim_vars = {'latitude': n_latitude, 'longitude': n_longitude, 'depth': n_depth,
                          'node': n_nodes, 'element': n_elements}
        self._grid_vars = {'nv': nv, 'nbe': nbe, 'longitude': lon_nodes, 'longitude_c': lon_elements,
                           'latitude': lat_nodes, 'latitude_c': lat_elements, 'depth': depth, 'h': h,
                           'mask': land_sea_mask_elements, 'mask_nodes': land_sea_mask_nodes,
                           'permutation': permutation, 'trim_first_latitude': trim_first_latitude,
                           'trim_last_latitude': trim_last_latitude}

        # Set dimensions
        zos_dimensions = ('time', 'latitude', 'longitude')
        uvwts_dimensions = ('time', 'depth', 'latitude', 'longitude')
        self._time_dep_var_dimensions = {'zos': zos_dimensions, 'uo': uvwts_dimensions, 'vo': uvwts_dimensions,
                                         'wo': uvwts_dimensions, 'thetao': uvwts_dimensions, 'so': uvwts_dimensions}

        # Store in dictionaries
        self._time_dep_vars_last = {'zos': zeta_t0, 'uo': uvw_t0, 'vo': uvw_t0, 'wo': uvw_t0, 'thetao': ts_t0,
                                    'so': ts_t0}

        self._time_dep_vars_next = {'zos': zeta_t1, 'uo': uvw_t1, 'vo': uvw_t1, 'wo': uvw_t1, 'thetao': ts_t0,
                                    'so': ts_t1}

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


class ArawawaADataReader_test(TestCase):

    def setUp(self):
        self.deg_to_radians = np.radians(1)

        # Create config
        config = configparser.ConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')
        config.add_section("SIMULATION")
        config.set('SIMULATION', 'time_direction', 'forward')
        config.set('SIMULATION', 'surface_only', 'False')
        config.add_section("OCEAN_CIRCULATION_MODEL")
        config.set('OCEAN_CIRCULATION_MODEL', 'time_dim_name', 'time')
        config.set('OCEAN_CIRCULATION_MODEL', 'depth_dim_name', 'depth')
        config.set('OCEAN_CIRCULATION_MODEL', 'latitude_dim_name', 'latitude')
        config.set('OCEAN_CIRCULATION_MODEL', 'longitude_dim_name', 'longitude')
        config.set('OCEAN_CIRCULATION_MODEL', 'time_var_name', 'time')
        config.set('OCEAN_CIRCULATION_MODEL', 'uo_var_name', 'uo')
        config.set('OCEAN_CIRCULATION_MODEL', 'vo_var_name', 'vo')
        config.set('OCEAN_CIRCULATION_MODEL', 'wo_var_name', 'wo')
        config.set('OCEAN_CIRCULATION_MODEL', 'zos_var_name', 'zos')
        config.set('OCEAN_CIRCULATION_MODEL', 'thetao_var_name', 'thetao')
        config.set('OCEAN_CIRCULATION_MODEL', 'so_var_name', 'so')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_is_wet', 'True')
        config.set('OCEAN_CIRCULATION_MODEL', 'coordinate_system', 'geographic')
        config.add_section("OUTPUT")
        config.set('OUTPUT', 'environmental_variables', 'thetao, so')

        # Create mediator
        mediator = MockArakawaAMediator()
        
        # Create data reader
        self.data_reader = ArakawaADataReader(config, mediator)
        
        # Read in data
        datetime_start = datetime.datetime(2000, 1, 1)  # Arbitrary start time, ignored by mock mediator
        datetime_end = datetime.datetime(2000, 1, 1)  # Arbitrary end time, ignored by mock mediator
        self.data_reader.setup_data_access(datetime_start, datetime_end)

    def tearDown(self):
        del(self.data_reader)

    def test_find_host_using_global_search(self):
        particle = ParticleSmartPtr(x1=self.deg_to_radians*1.666666667, x2=self.deg_to_radians*11.666666667)
        flag = self.data_reader.find_host_using_global_search_wrapper(particle)
        test.assert_equal(particle.get_host_horizontal_elem('arakawa_a'), 0)
        test.assert_equal(flag, 0)

    def test_find_host_when_particle_is_in_the_domain(self):
        particle_old = ParticleSmartPtr(x1=self.deg_to_radians*2.666666667, x2=self.deg_to_radians*11.333333333,
                                        host_elements={'arakawa_a': 7})
        particle_new = ParticleSmartPtr(x1=self.deg_to_radians*2.333333333, x2=self.deg_to_radians*11.6666666667,
                                        host_elements={'arakawa_a': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.get_host_horizontal_elem('arakawa_a'), 6)

    def test_find_host_when_particle_has_crossed_a_land_boundary(self):
        particle_old = ParticleSmartPtr(x1=self.deg_to_radians*2.333333333, x2=self.deg_to_radians*12.666666667,
                                        host_elements={'arakawa_a': 10})
        particle_new = ParticleSmartPtr(x1=self.deg_to_radians*2.333333333, x2=self.deg_to_radians*13.1,
                                        host_elements={'arakawa_a': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, -1)
        test.assert_equal(particle_new.get_host_horizontal_elem('arakawa_a'), 10)

    def test_get_zmin(self):
        x1 = self.deg_to_radians*2.0
        x2 = self.deg_to_radians*12.0
        host_elements = {'arakawa_a': 0}

        time = 0.0

        particle = ParticleSmartPtr(x1=x1, x2=x2, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        bathy = self.data_reader.get_zmin_wrapper(time, particle)
        test.assert_almost_equal(bathy, -10.0)

    def test_get_zmax(self):
        x1 = self.deg_to_radians * 2.0
        x2 = self.deg_to_radians * 12.0
        host_elements = {'arakawa_a': 0}

        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        zeta = self.data_reader.get_zmax_wrapper(time, particle)
        test.assert_almost_equal(zeta, 0.0)

        time = 1800.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        zeta = self.data_reader.get_zmax_wrapper(time, particle)
        test.assert_almost_equal(zeta, 0.0)

    def test_set_vertical_grid_vars_for_a_particle_on_the_sea_surface(self):
        time = 0.0
        x1 = self.deg_to_radians * 2.0
        x2 = self.deg_to_radians * 12.0
        x3 = 0.0
        host_elements = {'arakawa_a': 0}

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 0)
        test.assert_equal(particle.in_vertical_boundary_layer, False)
        test.assert_almost_equal(particle.omega_interfaces, 1.0)

    def test_set_vertical_grid_vars_for_a_particle_on_the_sea_floor(self):
        time = 0.0
        x1 = self.deg_to_radians * 2.0
        x2 = self.deg_to_radians * 12.0
        x3 = -10.
        host_elements = {'arakawa_a': 0}

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 0)
        test.assert_equal(particle.in_vertical_boundary_layer, False)
        test.assert_almost_equal(particle.omega_interfaces, 0.0)

    def test_set_vertical_grid_vars_for_a_particle_in_the_surface_layer(self):
        time = 0.0
        x1 = self.deg_to_radians * 2.0
        x2 = self.deg_to_radians * 12.0
        x3 = -5.0  # 1 m below the moving free surface
        host_elements = {'arakawa_a': 0}

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 0)
        test.assert_equal(particle.in_vertical_boundary_layer, False)
        test.assert_almost_equal(particle.omega_interfaces, 0.5)

    def test_get_velocity_in_surface_layer(self):
        x1 = self.deg_to_radians * 2.0
        x2 = self.deg_to_radians * 12.0
        host_elements = {'arakawa_a': 0}

        # Test #1
        x3 = 0.0 # Surface
        time = 0.0

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        test.assert_equal(flag, 0)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_array_almost_equal(vel, [1., 1., 1.])

        # Test #2
        x3 = -5.0  # Half way down the surface layer
        time = 1800.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [0.5, 0.5, 0.5])

    def test_get_velocity_in_surface_layer_in_a_boundary_element(self):
        x1 = self.deg_to_radians * 1.333333333
        x2 = self.deg_to_radians * 12.333333333
        host_elements = {'arakawa_a': 4}

        # Test #1
        x3 = 0.0 # Surface
        time = 0.0

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        test.assert_equal(flag, 0)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_array_almost_equal(vel, [1., 1., 1.])

        # Test #2
        x3 = -5.0  # Half way down the surface layer
        time = 1800.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [0.5, 0.5, 0.5])

    def test_get_velocity_in_surface_layer_on_a_boundary(self):
        x1 = self.deg_to_radians * 2.0
        x2 = self.deg_to_radians * 13.0
        host_elements = {'arakawa_a': 5}

        # Test #1
        x3 = 0.0 # Surface
        time = 0.0

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        test.assert_equal(flag, 0)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_array_almost_equal(vel, [1., 1., 1.])

        # Test #2
        x3 = -5.0  # Half way down the surface layer
        time = 1800.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [0.5, 0.5, 0.5])

    def test_get_thetao(self):
        x1 = self.deg_to_radians * 2.0
        x2 = self.deg_to_radians * 12.0
        host_elements = {'arakawa_a': 0}

        x3 = -5.0
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        thetao = self.data_reader.get_environmental_variable_wrapper('thetao', time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(thetao,  0.5)

    def test_get_so(self):
        x1 = self.deg_to_radians * 2.0
        x2 = self.deg_to_radians * 12.0
        host_elements = {'arakawa_a': 0}

        x3 = -5.
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        so = self.data_reader.get_environmental_variable_wrapper('so', time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(so,  0.5)

    # def test_get_vertical_eddy_diffusivity(self):
    #     x1 = self.deg_to_radians * 1.3333333333
    #     x2 = self.deg_to_radians * 1.6666666667
    #     host = 0
    #
    #     x3 = -0.2
    #     time = 0.0
    #     particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host=host)
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #     diffusivity = self.data_reader.get_vertical_eddy_diffusivity_wrapper(time, particle)
    #     test.assert_equal(flag, 0)
    #     test.assert_almost_equal(diffusivity,  0.005)
    #
    # def test_get_vertical_eddy_diffusivity_derivative(self):
    #     x1 = self.deg_to_radians * 1.3333333333
    #     x2 = self.deg_to_radians * 1.6666666667
    #     host = 0
    #
    #     x3 = -0.2
    #     time = 0.0
    #     particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host=host)
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #     diffusivity_gradient = self.data_reader.get_vertical_eddy_diffusivity_derivative_wrapper(time, particle)
    #     test.assert_equal(flag, 0)
    #     test.assert_almost_equal(diffusivity_gradient, -0.0026042)
    #
    # def test_get_horizontal_eddy_viscosity(self):
    #     x1 = self.deg_to_radians * 1.3333333333
    #     x2 = self.deg_to_radians * 1.6666666667
    #     host = 0
    #
    #     x3 = -0.1
    #     time = 0.0
    #     particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host=host)
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #     viscosity = self.data_reader.get_horizontal_eddy_viscosity_wrapper(time, particle)
    #     test.assert_equal(flag, 0)
    #     test.assert_almost_equal(viscosity,  0.01)
    #
    # def test_get_horizontal_eddy_viscosity_derivative(self):
    #     x1 = self.deg_to_radians * 1.3333333333
    #     x2 = self.deg_to_radians * 1.6666666667
    #     host = 0
    #
    #     x3 = -0.1
    #     time = 0.0
    #     particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host=host)
    #     self.data_reader.set_local_coordinates_wrapper(particle)
    #     flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
    #
    #     Ah_prime = np.empty(2, dtype=DTYPE_FLOAT)
    #     self.data_reader.get_horizontal_eddy_viscosity_derivative_wrapper(time, particle, Ah_prime)
    #     test.assert_equal(flag, 0)
    #     test.assert_array_almost_equal(Ah_prime, [0.0, 0.0])
    #
    def test_element_is_wet(self):
        x1 = self.deg_to_radians * 2.3333333333
        x2 = self.deg_to_radians * 11.6666666667
        host_elements = {'arakawa_a': 6}

        time = 0.0

        particle = ParticleSmartPtr(x1=x1, x2=x2, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        status = self.data_reader.is_wet_wrapper(time, particle)
        test.assert_equal(status, 1)

    def test_element_is_dry(self):
        x1 = self.deg_to_radians * 1.6666666667
        x2 = self.deg_to_radians * 11.6666666667
        host_elements={'arakawa_a': 0}

        time = 0.0

        particle = ParticleSmartPtr(x1=x1, x2=x2, host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)
        status = self.data_reader.is_wet_wrapper(time, particle)
        test.assert_equal(status, 0)


class ArakawaAReflectingHorizBoundaryCondition_test(TestCase):

    def setUp(self):
        self.deg_to_radians = np.radians(1)

        # Create config
        config = configparser.ConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')
        config.add_section("SIMULATION")
        config.set('SIMULATION', 'time_direction', 'forward')
        config.set('SIMULATION', 'surface_only', 'False')
        config.add_section("OCEAN_CIRCULATION_MODEL")
        config.set('OCEAN_CIRCULATION_MODEL', 'time_dim_name', 'time')
        config.set('OCEAN_CIRCULATION_MODEL', 'depth_dim_name', 'depth')
        config.set('OCEAN_CIRCULATION_MODEL', 'latitude_dim_name', 'latitude')
        config.set('OCEAN_CIRCULATION_MODEL', 'longitude_dim_name', 'longitude')
        config.set('OCEAN_CIRCULATION_MODEL', 'time_var_name', 'time')
        config.set('OCEAN_CIRCULATION_MODEL', 'uo_var_name', 'uo')
        config.set('OCEAN_CIRCULATION_MODEL', 'vo_var_name', 'vo')
        config.set('OCEAN_CIRCULATION_MODEL', 'wo_var_name', 'wo')
        config.set('OCEAN_CIRCULATION_MODEL', 'zos_var_name', 'zos')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_is_wet', 'True')
        config.set('OCEAN_CIRCULATION_MODEL', 'coordinate_system', 'geographic')
        config.add_section("OUTPUT")

        # Create mediator
        mediator = MockArakawaAMediator()

        # Create data reader
        self.data_reader = ArakawaADataReader(config, mediator)

        # Read in data
        datetime_start = datetime.datetime(2000, 1, 1)  # Arbitrary start time, ignored by mock mediator
        datetime_end = datetime.datetime(2000, 1, 1)  # Arbitrary end time, ignored by mock mediator
        self.data_reader.setup_data_access(datetime_start, datetime_end)

        # Boundary condition calculator
        self.horiz_boundary_condition_calculator = RefHorizBoundaryConditionCalculator()

    def tearDown(self):
        del (self.data_reader)

    def test_reflect_particle_on_a_normal_trajectory(self):
        particle_old = ParticleSmartPtr(x1=1.99*self.deg_to_radians, x2=12.8*self.deg_to_radians, host_elements={'arakawa_a': 5})
        particle_new = ParticleSmartPtr(x1=1.99*self.deg_to_radians, x2=13.1*self.deg_to_radians, host_elements={'arakawa_a': 5})
        flag = self.horiz_boundary_condition_calculator.apply_wrapper(self.data_reader, particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(particle_new.x1, 1.99*self.deg_to_radians, decimal=2)
        test.assert_almost_equal(particle_new.x2, 12.9*self.deg_to_radians, decimal=2)
        test.assert_equal(particle_new.get_host_horizontal_elem('arakawa_a'), 5)

    # def test_reflect_particle_on_an_angled_trajectory(self):
    #     particle_old = ParticleSmartPtr(x1=1.4 - self.xmin, x2=1.1 - self.ymin, host_elements={'fvcom': 1})
    #     particle_new = ParticleSmartPtr(x1=1.6 - self.xmin, x2=0.9 - self.ymin, host_elements={'fvcom': 1})
    #     flag = self.horiz_boundary_condition_calculator.apply_wrapper(self.data_reader, particle_old, particle_new)
    #     test.assert_equal(flag, 0)
    #     test.assert_equal(particle_new.x1 + self.xmin, 1.6)
    #     test.assert_equal(particle_new.x2 + self.ymin, 1.1)
    #     test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 1)
    #
    # def test_reflect_particle_that_sits_on_the_boundary(self):
    #     particle_old = ParticleSmartPtr(x1=1.5 - self.xmin, x2=1.0 - self.ymin, host_elements={'fvcom': 1})
    #     particle_new = ParticleSmartPtr(x1=1.6 - self.xmin, x2=0.9 - self.ymin, host_elements={'fvcom': 1})
    #     flag = self.horiz_boundary_condition_calculator.apply_wrapper(self.data_reader, particle_old, particle_new)
    #     test.assert_equal(flag, 0)
    #     test.assert_equal(particle_new.x1 + self.xmin, 1.6)
    #     test.assert_equal(particle_new.x2 + self.ymin, 1.1)
    #     test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 1)
    #
    # def test_reflect_particle_that_undergoes_a_double_reflection_while_on_a_southeast_trajectory(self):
    #     particle_old = ParticleSmartPtr(x1=1.8 - self.xmin, x2=1.1 - self.ymin, host_elements={'fvcom': 1})
    #     particle_new = ParticleSmartPtr(x1=2.1 - self.xmin, x2=0.8 - self.ymin, host_elements={'fvcom': 1})
    #     flag = self.horiz_boundary_condition_calculator.apply_wrapper(self.data_reader, particle_old, particle_new)
    #     test.assert_equal(flag, 0)
    #     test.assert_equal(particle_new.x1 + self.xmin, 1.9)
    #     test.assert_equal(particle_new.x2 + self.ymin, 1.2)
    #     test.assert_equal(particle_new.get_host_horizontal_elem('fvcom'), 1)
