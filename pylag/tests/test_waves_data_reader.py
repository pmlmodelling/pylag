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

from pylag.waves_data_reader import WavesDataReader
from pylag.particle_cpp_wrapper import ParticleSmartPtr

from pylag.mediator import Mediator
from pylag import grid_metrics as gm


class MockWavesMediator(Mediator):
    """ Test mediator for wave data

    We first define a structured 3x3x3 grid with lat and lon dimensions, in that
    order. From this we construct the unstructured grid using the approach
    adopted within create_arakawa_a_grid_metrics_file() in grid_metrics.py.
    This gives the unstructured grid variables (lat, lon, lat_c, lon_c) to be
    read by PyLag. We construct the grid without a mask. Next, we create the
    time dependent variables (vsdx, vsdy). These are kept on their native
    structured grid - it is down to PyLag to correctly interpret and handle
    these, with accompanying unit tests put in place to ensure this is so.
    """
    def __init__(self):
        self.deg_to_radians = np.radians(1)

        # Basic grid (4 x 3).
        latitude = np.array([11., 12., 13., 14], dtype=DTYPE_FLOAT)
        longitude = np.array([1., 2., 3.], dtype=DTYPE_FLOAT)

        # Save original grid dimensions
        n_latitude = latitude.shape[0]
        n_longitude = longitude.shape[0]

        # vsdx and vsdy (time = 0) [lat, lon].
        vsdx_t0 = np.ma.masked_array([[4., 4., 4.], [3., 3., 3.],
                                      [2., 2., 2.], [1., 1., 1.]],
                                     dtype=DTYPE_FLOAT)
        vsdy_t0 = np.ma.masked_array([[14., 14., 14.], [13., 13., 13.],
                                      [12., 12., 12.], [11., 11., 11.]],
                                     dtype=DTYPE_FLOAT)
        vsdx_t1 = np.ma.copy(vsdx_t0)
        vsdy_t1 = np.ma.copy(vsdy_t0)

        # Trim latitudes
        trim_first_latitude = np.array([False])
        trim_last_latitude = np.array([False])

        # Form the unstructured grid
        lon2d, lat2d = np.meshgrid(longitude[:], latitude[:], indexing='ij')
        points = np.array([lon2d.flatten(order='C'),
                           lat2d.flatten(order='C')]).T

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
        lon_elements, lat_elements = \
            gm.compute_element_midpoints_in_geographic_coordinates(nv,
                                                                   lon_nodes,
                                                                   lat_nodes)

        # Save dummy array of element areas
        areas = np.ones(n_elements, dtype=DTYPE_FLOAT)

        # Transpose arrays
        nv = nv.T
        nbe = nbe.T

        # Flag open boundaries with -2 flag
        nbe[np.asarray(nbe == -1).nonzero()] = -2

        # Add to grid dimensions and variables
        self._dim_vars = {'latitude': n_latitude, 'longitude': n_longitude,
                          'node': n_nodes, 'element': n_elements}
        self._grid_vars = {'nv': nv, 'nbe': nbe, 'longitude': lon_nodes,
                           'longitude_c': lon_elements,
                           'latitude': lat_nodes, 'latitude_c': lat_elements,
                           'permutation': permutation,
                           'trim_first_latitude': trim_first_latitude,
                           'trim_last_latitude': trim_last_latitude,
                           'area': areas}

        # Set dimensions
        uv_dimensions = ('time', 'latitude', 'longitude')
        self._time_dep_var_dimensions = {'vsdx': uv_dimensions,
                                         'vsdy': uv_dimensions}

        # Store in dictionaries
        self._time_dep_vars_last = {'vsdx': vsdx_t0, 'vsdy': vsdy_t0}
        self._time_dep_vars_next = {'vsdx': vsdx_t1, 'vsdy': vsdy_t1}

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
        return np.ascontiguousarray(self._grid_vars[var_name][:].astype(var_type))

    def get_variable_dimensions(self, var_name, include_time=True):
        if include_time:
            return self._time_dep_var_dimensions[var_name]
        else:
            return self._time_dep_var_dimensions[var_name][1:]

    def get_variable_shape(self, var_name, include_time=True):
        if include_time:
            return self._time_dep_vars_last[var_name].shape
        else:
            return self._time_dep_vars_last[var_name].shape[1:]

    def get_time_dependent_variable_at_last_time_index(self, var_name,
                                                       var_dims, var_type):
        var = self._time_dep_vars_last[var_name][:].astype(var_type)

        if np.ma.isMaskedArray(var):
            var = var.data

        return np.ascontiguousarray(var)

    def get_time_dependent_variable_at_next_time_index(self, var_name,
                                                       var_dims, var_type):
        var = self._time_dep_vars_next[var_name][:].astype(var_type)

        if np.ma.isMaskedArray(var):
            var = var.data

        return np.ascontiguousarray(var)

    def get_mask_at_last_time_index(self, var_name, var_dims):
        var = self._time_dep_vars_last[var_name][:]

        if np.ma.isMaskedArray(var):
            return np.ascontiguousarray(var.mask.astype(DTYPE_INT))

        raise RuntimeError(f'Variable {var_name} is not a masked array.')

    def get_mask_at_next_time_index(self, var_name, var_dims):
        var = self._time_dep_vars_next[var_name][:]

        if np.ma.isMaskedArray(var):
            return np.ascontiguousarray(var.mask.astype(DTYPE_INT))

        raise RuntimeError(f'Variable {var_name} is not a masked array.')


class WavesDataReader_test(TestCase):

    def setUp(self):
        self.deg_to_radians = np.radians(1)

        # Create config
        config = configparser.ConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')
        config.add_section("SIMULATION")
        config.set('SIMULATION', 'time_direction', 'forward')
        config.set('SIMULATION', 'surface_only', 'False')
        config.set('SIMULATION', 'coordinate_system', 'geographic')
        config.add_section("WAVE_DATA")
        config.set('WAVE_DATA', 'time_dim_name', 'time')
        config.set('WAVE_DATA', 'latitude_dim_name', 'latitude')
        config.set('WAVE_DATA', 'longitude_dim_name', 'longitude')
        config.set('WAVE_DATA', 'time_var_name', 'time')
        config.set('WAVE_DATA', 'vsdx_var_name', 'vsdx')
        config.set('WAVE_DATA', 'vsdy_var_name', 'vsdy')

        # Create mediator
        mediator = MockWavesMediator()
        
        # Create data reader
        self.data_reader = WavesDataReader(config, mediator)
        
        # Read in data
        datetime_start = datetime.datetime(2000, 1, 1)  # Dummy value
        datetime_end = datetime.datetime(2000, 1, 1)  # Dummy value
        self.data_reader.setup_data_access(datetime_start, datetime_end)

    def tearDown(self):
        del(self.data_reader)

    def test_find_host_using_global_search(self):
        particle = ParticleSmartPtr(x1=self.deg_to_radians*1.666666667,
                                    x2=self.deg_to_radians*11.666666667)
        flag = self.data_reader.find_host_using_global_search_wrapper(particle)
        test.assert_equal(particle.get_host_horizontal_elem('waves'), 0)
        test.assert_equal(flag, 0)

    def test_find_host_when_particle_is_in_the_domain(self):
        particle_old = ParticleSmartPtr(x1=self.deg_to_radians*2.666666667,
                                        x2=self.deg_to_radians*11.333333333,
                                        host_elements={'waves': 7})
        particle_new = ParticleSmartPtr(x1=self.deg_to_radians*2.333333333,
                                        x2=self.deg_to_radians*11.6666666667,
                                        host_elements={'waves': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.get_host_horizontal_elem('waves'),
                          6)

    def test_find_host_when_particle_has_moved_outside_of_the_domain(self):
        particle_old = ParticleSmartPtr(x1=self.deg_to_radians*2.333333333,
                                        x2=self.deg_to_radians*12.666666667,
                                        host_elements={'waves': 10})
        particle_new = ParticleSmartPtr(x1=self.deg_to_radians*2.333333333,
                                        x2=self.deg_to_radians*14.1,
                                        host_elements={'waves': -999})
        flag = self.data_reader.find_host_wrapper(particle_old, particle_new)
        test.assert_equal(flag, -2)

    def test_get_surface_stokes_drift_velocity(self):
        x1 = self.deg_to_radians * 2.0
        x2 = self.deg_to_radians * 12.0
        host_elements = {'waves': 0}

        # Test #1
        x3 = 0.0
        time = 0.0

        particle = ParticleSmartPtr(x1=x1, x2=x2, x3=x3,
                                    host_elements=host_elements)
        self.data_reader.set_local_coordinates_wrapper(particle)

        vel = self.data_reader.get_surface_stokes_drift_velocity_wrapper(time,
                                                                         particle)
        test.assert_array_almost_equal(vel, [3., 13.])
