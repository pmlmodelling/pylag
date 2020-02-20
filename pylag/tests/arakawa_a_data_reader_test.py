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
#from pylag.boundary_conditions import RefHorizBoundaryConditionCalculator
#from pylag.boundary_conditions import RefVertBoundaryConditionCalculator
from pylag.particle import ParticleSmartPtr
from pylag import cwrappers

from pylag.mediator import Mediator
from pylag.grid_metrics import sort_adjacency_array


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
        # Basic grid (3 x 3 x 4).
        latitude = np.array([11., 12., 13.], dtype=float)
        longitude = np.array([1., 2., 3.], dtype=float)
        depth = np.array([0., 10., 20., 30.], dtype=float)

        # Save original grid dimensions
        n_latitude = latitude.shape[0]
        n_longitude = longitude.shape[0]
        n_depth = depth.shape[0]

        # Bathymetry [lat, lon]
        h = np.array([[25., 25., 25.], [10., 10., 10.], [999., 999., 999.]])
        h = np.moveaxis(h, 1, 0)  # Move to [lon, lat]
        h = h.reshape(np.prod(h.shape), order='C')

        # Mask [depth, lat, lon]. 1 is sea, 0 land. Note the last depth level is masked everywhere.
        mask = np.array([[[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                         [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                         [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=int)

        # Switch the mask convention to that which PyLag anticipates. i.e. 1 is a masked point, 0 a non-masked point.
        mask = 1 - mask

        # Separately save the surface mask at nodes, This is taken as the land sea mask.
        land_sea_mask = mask[0, :, :]
        land_sea_mask_nodes = np.moveaxis(land_sea_mask, 0, 1)  # Move to [lon, lat]
        land_sea_mask_nodes = land_sea_mask_nodes.reshape(np.prod(land_sea_mask_nodes.shape), order='C')

        # Zeta (time = 0) [lat, lon]
        zeta_t0 = np.ma.masked_array([[1., 1., 1.], [0., 0., 0.], [999., 999., 999.]], mask=land_sea_mask, dtype=float)
        zeta_t1 = np.ma.copy(zeta_t0)

        # u/v/w (time = 0) [depth, lat, lon]. Include mask.
        uvw_t0 = np.ma.masked_array([[[2., 2., 2.], [1., 1., 1.], [999., 999., 999.]],
                                     [[1., 1., 1.], [0., 0., 0.], [999., 999., 999.]],
                                     [[0., 0., 0.], [999., 999., 999.], [999., 999., 999.]],
                                     [[999., 999., 999.], [999., 999., 999.], [999., 999., 999.]]],
                                    mask=mask, dtype=float)
        uvw_t1 = np.ma.copy(uvw_t0)

        # t/s (time = 0) [depth, lat, lon]. Include mask.
        ts_t0 = np.ma.masked_array([[[2., 2., 2.], [1., 1., 1.], [999., 999., 999.]],
                                     [[1., 1., 1.], [0., 0., 0.], [999., 999., 999.]],
                                     [[0., 0., 0.], [999., 999., 999.], [999., 999., 999.]],
                                     [[999., 999., 999.], [999., 999., 999.], [999., 999., 999.]]],
                                    mask=mask, dtype=float)
        ts_t1 = np.ma.copy(ts_t0)

        # Form the unstructured grid
        lon2d, lat2d = np.meshgrid(longitude[:], latitude[:], indexing='ij')
        points = np.array([lon2d.flatten(order='C'), lat2d.flatten(order='C')]).T

        # Save lon and lat points at nodes
        lon_nodes = points[:, 0]
        lat_nodes = points[:, 1]
        n_nodes = points.shape[0]

        # Create the Triangulation
        tri = Delaunay(points)

        # Save simplices
        #   - Flip to reverse ordering, as expected by PyLag
        #   - Transpose to give it the dimension ordering expected by PyLag
        nv = np.flip(tri.simplices.copy(), axis=1).T
        n_elements = nv.shape[1]

        # Save neighbours
        #   - Transpose to give it the dimension ordering expected by PyLag
        #   - Sort to ensure match with nv
        nbe = tri.neighbors.T
        nbe = sort_adjacency_array(nv, nbe)

        # Save lon and lat points at element centres
        lon_elements = np.empty(n_elements, dtype=float)
        lat_elements = np.empty(n_elements, dtype=float)
        for i, element in enumerate(range(n_elements)):
            lon_elements[i] = lon_nodes[(nv[:, element])].mean()
            lat_elements[i] = lat_nodes[(nv[:, element])].mean()

        # Generate the land-sea mask for elements
        land_sea_mask_elements = np.empty(n_elements, dtype=int)
        for i in range(n_elements):
            element_nodes = nv[:, i]
            land_sea_mask_elements[i] = 1 if np.any(land_sea_mask_nodes[(element_nodes)] == 1) else 0

        # Flag open boundaries with -2 flag
        nbe[np.where(nbe == -1)] = -2

        # Flag land boundaries with -1 flag
        for i, msk in enumerate(land_sea_mask_elements):
            if msk == 1:
                nbe[np.where(nbe == i)] = -1

        # Add to grid dimensions and variables
        self._dim_vars = {'latitude': n_latitude, 'longitude': n_longitude, 'depth': n_depth,
                          'node': n_nodes, 'element': n_elements}
        self._grid_vars = {'nv': nv, 'nbe': nbe, 'longitude': lon_nodes, 'longitude_c': lon_elements,
                           'latitude': lat_nodes, 'latitude_c': lat_elements, 'depth': depth, 'h': h,
                           'mask': land_sea_mask_elements}

        # Store in dictionaries
        self._time_dep_vars_last = {'zeta': zeta_t0, 'uo': uvw_t0, 'vo': uvw_t0, 'wo': uvw_t0, 'thetao': ts_t0,
                                    'so': ts_t0}

        self._time_dep_vars_next = {'zeta': zeta_t1, 'uo': uvw_t1, 'vo': uvw_t1, 'wo': uvw_t1, 'thetao': ts_t0,
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
        # Create config
        config = configparser.SafeConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')
        config.add_section("SIMULATION")
        config.set('SIMULATION', 'time_direction', 'forward')
        config.add_section("OCEAN_CIRCULATION_MODEL")
        config.set('OCEAN_CIRCULATION_MODEL', 'has_w', 'True')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_Kh', 'False')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_Ah', 'False')
        config.set('OCEAN_CIRCULATION_MODEL', 'has_is_wet', 'False')
        config.set('OCEAN_CIRCULATION_MODEL', 'coordinate_system', 'spherical')
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

        # Grid offsets
        self.xmin = self.data_reader.get_xmin()
        self.ymin = self.data_reader.get_ymin()

    def tearDown(self):
        del(self.data_reader)

    def test_find_host_using_global_search(self):
        particle = ParticleSmartPtr(x1=1.666666667-self.xmin, x2=11.666666667-self.ymin)
        flag = self.data_reader.find_host_using_global_search_wrapper(particle)
        test.assert_equal(particle.host_horizontal_elem, 0)
        test.assert_equal(flag, 0)

    def test_find_host_when_particle_is_in_the_domain(self):
        particle_old = ParticleSmartPtr(x1=2.666666667-self.xmin, x2=11.333333333-self.ymin, host=3)
        particle_new = ParticleSmartPtr(x1=2.333333333-self.xmin, x2=11.6666666667-self.ymin)
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.host_horizontal_elem, 2)

    def test_find_host_when_particle_has_crossed_a_land_boundary(self):
        particle_old = ParticleSmartPtr(x1=2.333333333-self.xmin, x2=11.666666667-self.ymin, host=2)
        particle_new = ParticleSmartPtr(x1=2.333333333-self.xmin, x2=12.1-self.ymin)
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, -1)
        test.assert_equal(particle_new.host_horizontal_elem, 2)

    def test_get_zmin(self):
        x1 = 2.333333333-self.xmin
        x2 = 11.66666667-self.ymin
        host = 2

        time = 0.0

        particle = ParticleSmartPtr(x1=x1, x2=x2, host=host)
        self.data_reader.set_local_coordinates_wrapper(particle)
        bathy = self.data_reader.get_zmin_wrapper(time, particle)
        test.assert_almost_equal(bathy, -15.0)

    def test_get_zmax(self):
        x1 = 2.3333333333-self.xmin
        x2 = 11.6666666667-self.ymin
        host = 2

        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, host=host)
        self.data_reader.set_local_coordinates_wrapper(particle)
        zeta = self.data_reader.get_zmax_wrapper(time, particle)
        test.assert_almost_equal(zeta, 0.333333333)

        time = 1800.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, host=host)
        self.data_reader.set_local_coordinates_wrapper(particle)
        zeta = self.data_reader.get_zmax_wrapper(time, particle)
        test.assert_almost_equal(zeta, 0.333333333)

    def test_set_vertical_grid_vars_for_a_particle_on_the_sea_surface(self):
        time = 0.0
        x1 = 2.3333333333-self.xmin
        x2 = 11.6666666667-self.ymin
        x3 = 0.333333333
        host = 2

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host=host)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 0)
        test.assert_equal(particle.in_vertical_boundary_layer, False)
        test.assert_almost_equal(particle.omega_interfaces, 1.0)

    def test_set_vertical_grid_vars_for_a_particle_on_the_sea_floor(self):
        time = 0.0
        x1 = 2.3333333333-self.xmin
        x2 = 11.6666666667-self.ymin
        x3 = -15.
        host = 2

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host=host)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 1)
        test.assert_equal(particle.in_vertical_boundary_layer, True)
        test.assert_almost_equal(particle.omega_interfaces, 0.466666667)

    def test_set_vertical_grid_vars_for_a_particle_in_the_surface_layer(self):
        time = 0.0
        x1 = 2.3333333333-self.xmin
        x2 = 11.6666666667-self.ymin
        x3 = -0.666666667  # 1 m below the moving free surface
        host = 2

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host=host)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        test.assert_equal(flag, 0)
        test.assert_equal(particle.k_layer, 0)
        test.assert_equal(particle.in_vertical_boundary_layer, False)
        test.assert_almost_equal(particle.omega_interfaces, 0.9)

    def test_get_velocity_in_surface_layer(self):
        x1 = 2.3333333333-self.xmin
        x2 = 11.6666666667-self.ymin
        host = 2

        # Test #1
        x3 = 0.333333333
        time = 0.0

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host=host)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        test.assert_equal(flag, 0)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_array_almost_equal(vel, [1.333333333, 1.333333333, 1.333333333])

        # Test #2
        x3 = -4.6666666667 # Half way down the middle layer
        time = 1800.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host=host)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [0.8333333333, 0.8333333333, 0.8333333333])

    def test_get_velocity_in_middle_layer(self):
        x1 = 2.3333333333-self.xmin
        x2 = 11.6666666667-self.ymin
        host = 2

        # Test #1
        x3 = -14.666666667  # Half way down middle layer and 0.333 m above the sea floor
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host=host)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)

        vel = np.empty(3, dtype=DTYPE_FLOAT)
        self.data_reader.get_velocity_wrapper(time, particle, vel)
        test.assert_equal(flag, 0)
        test.assert_array_almost_equal(vel, [0.333333333, 0.333333333, 0.333333333])

    def test_get_thetao(self):
        x1 = 2.3333333333-self.xmin
        x2 = 11.6666666667-self.ymin
        host = 2

        x3 = 0.333333333
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host=host)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        thetao = self.data_reader.get_environmental_variable_wrapper('thetao', time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(thetao,  1.333333333)

    def test_get_so(self):
        x1 = 2.3333333333-self.xmin
        x2 = 11.6666666667-self.ymin
        host = 2

        x3 = 0.333333333
        time = 0.0
        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3, host=host)
        self.data_reader.set_local_coordinates_wrapper(particle)
        flag = self.data_reader.set_vertical_grid_vars_wrapper(time, particle)
        so = self.data_reader.get_environmental_variable_wrapper('so', time, particle)
        test.assert_equal(flag, 0)
        test.assert_almost_equal(so,  1.333333333)

    # def test_get_vertical_eddy_diffusivity(self):
    #     x1 = 1.3333333333-self.xmin
    #     x2 = 1.6666666667-self.ymin
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
    #     x1 = 1.3333333333-self.xmin
    #     x2 = 1.6666666667-self.ymin
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
    #     x1 = 1.3333333333-self.xmin
    #     x2 = 1.6666666667-self.ymin
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
    #     x1 = 1.3333333333-self.xmin
    #     x2 = 1.6666666667-self.ymin
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
    # def test_element_is_wet(self):
    #     host = 0
    #     time = 0.0
    #
    #     status = self.data_reader.is_wet(time, host)
    #     test.assert_equal(status, 1)

