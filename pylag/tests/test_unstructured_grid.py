from unittest import TestCase
import numpy.testing as test
import numpy as np
from scipy.spatial import Delaunay

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT

from pylag.particle_cpp_wrapper import ParticleSmartPtr
from pylag.unstructured import UnstructuredCartesianGrid, UnstructuredGeographicGrid
from pylag import grid_metrics as gm

class UnstructuredCartesianGrid_test(TestCase):
    """ Unit tests for unstructured Cartesian grids

    The tests use the same grid as is used with the FVCOM data reader tests. Some
    of the tests are also the same, but here they don't go through FVCOM data reader.
    """

    def setUp(self):
        self.name = b'test_grid'
        self.n_elems = 5
        self.n_nodes = 7
        self.nv = np.ascontiguousarray(np.array([[4, 0, 2, 4, 3], [3, 1, 4, 5, 6], [1, 3, 1, 3, 0]], dtype=DTYPE_INT))
        self.nbe = np.ascontiguousarray(np.array([[1, 0, 0, -1, -1], [2, 4, -1, 0, 1], [3, -1, -1, -1, -1]], dtype=DTYPE_INT))
        self.x = np.ascontiguousarray(np.array([2.0, 1.0, 0.0, 2.0, 1.0, 1.5, 3.0], dtype=DTYPE_FLOAT))
        self.y = np.ascontiguousarray(np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0], dtype=DTYPE_FLOAT))
        self.xc = np.ascontiguousarray(np.array([1.3333333333, 1.6666666667, 0.6666666667, 1.5000000000, 2.3333333333], dtype=DTYPE_FLOAT))
        self.yc = np.ascontiguousarray(np.array([1.6666666667, 1.3333333333, 1.3333333333, 2.3333333333, 1.3333333333], dtype=DTYPE_FLOAT))
        self.mask_nodes = np.ascontiguousarray(np.array([0, 0, 0, 0, 0], dtype=DTYPE_INT))
        self.mask = np.ascontiguousarray(np.array([0, 0, 2, 2, 2], dtype=DTYPE_INT))

        # Create config
        config = configparser.ConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')

        # Create unstructured grid
        self.unstructured_grid = UnstructuredCartesianGrid(config, self.name, self.n_nodes, self.n_elems,
                                                           self.nv, self.nbe, self.x, self.y, self.xc,
                                                           self.yc, self.mask, self.mask_nodes)
        
    def tearDown(self):
        del self.unstructured_grid

    def test_find_host_using_global_search(self):
        particle = ParticleSmartPtr(x1=1.3333333333, x2=1.6666666667)
        flag = self.unstructured_grid.find_host_using_global_search_wrapper(particle)
        test.assert_equal(particle.get_host_horizontal_elem('test_grid'), 0)
        test.assert_equal(flag, 0)

    def test_find_host_using_global_search_when_a_particle_is_in_an_element_with_two_land_boundaries(self):
        particle = ParticleSmartPtr(x1=0.6666666667, x2=1.3333333333, host_elements={'test_grid': -1})
        flag = self.unstructured_grid.find_host_using_global_search_wrapper(particle)
        test.assert_equal(particle.get_host_horizontal_elem('test_grid'), -1)

    def test_get_boundary_intersection(self):
        particle_old = ParticleSmartPtr(x1=1.6666666667, x2=1.3333333333, host_elements={'test_grid': 1})
        particle_new = ParticleSmartPtr(x1=1.6666666667, x2=0.9, host_elements={'test_grid': 1})
        start_point, end_point, intersection = self.unstructured_grid.get_boundary_intersection_wrapper(particle_old, particle_new)
        test.assert_almost_equal(start_point[0], 2.0)
        test.assert_almost_equal(start_point[1], 1.0)
        test.assert_almost_equal(end_point[0], 1.0)
        test.assert_almost_equal(end_point[1], 1.0)
        test.assert_almost_equal(intersection[0], 1.6666666667)
        test.assert_almost_equal(intersection[1], 1.0)

    def test_get_boundary_intersection_when_a_particle_has_left_an_external_elements_edge(self):
        particle_old = ParticleSmartPtr(x1=1.5, x2=1.0, host_elements={'test_grid': 1})
        particle_new = ParticleSmartPtr(x1=1.5, x2=0.9, host_elements={'test_grid': 1})
        start_point, end_point, intersection = self.unstructured_grid.get_boundary_intersection_wrapper(particle_old, particle_new)
        test.assert_almost_equal(start_point[0], 2.0)
        test.assert_almost_equal(start_point[1], 1.0)
        test.assert_almost_equal(end_point[0], 1.0)
        test.assert_almost_equal(end_point[1], 1.0)
        test.assert_almost_equal(intersection[0], 1.5)
        test.assert_almost_equal(intersection[1], 1.0)

    def test_get_boundary_intersection_when_a_particle_has_moved_to_an_external_elements_edge(self):
        particle_old = ParticleSmartPtr(x1=1.5, x2=1.1, host_elements={'test_grid': 1})
        particle_new = ParticleSmartPtr(x1=1.5, x2=1.0, host_elements={'test_grid': 1})
        start_point, end_point, intersection = self.unstructured_grid.get_boundary_intersection_wrapper(particle_old, particle_new)
        test.assert_almost_equal(start_point[0], 2.0)
        test.assert_almost_equal(start_point[1], 1.0)
        test.assert_almost_equal(end_point[0], 1.0)
        test.assert_almost_equal(end_point[1], 1.0)
        test.assert_almost_equal(intersection[0], 1.5)
        test.assert_almost_equal(intersection[1], 1.0)

    def test_set_default_location(self):
        particle = ParticleSmartPtr(x1=1.5, x2=1.0, host_elements={'test_grid': 1})
        self.unstructured_grid.set_default_location_wrapper(particle)
        test.assert_almost_equal(particle.x1, 1.66666666667)
        test.assert_almost_equal(particle.x2, 1.33333333333)
        test.assert_array_almost_equal(particle.get_phi('test_grid'), [0.3333333333, 0.3333333333, 0.3333333333])

    def test_set_local_coordinates_when_a_particle_is_on_an_external_elements_side(self):
        particle = ParticleSmartPtr(x1=1.5, x2=1.0, host_elements={'test_grid': 1})
        self.unstructured_grid.set_local_coordinates_wrapper(particle)
        phi_min = np.min(np.array(particle.get_phi('test_grid'), dtype=DTYPE_FLOAT))
        test.assert_equal(np.abs(phi_min), 0.0)

    def test_get_phi_when_particle_is_at_an_elements_centre(self):
        x1 = 1.666666667
        x2 = 1.333333333
        host = 1
        phi = self.unstructured_grid.get_phi(x1, x2, host)
        test.assert_array_almost_equal(phi, [1./3., 1./3., 1./3.])

    def test_get_phi_when_particle_is_at_an_elements_vertex(self):
        x1 = 2.0
        x2 = 1.0
        host = 1
        phi = self.unstructured_grid.get_phi(x1, x2, host)
        test.assert_array_almost_equal(phi, [1., 0., 0.])

    def test_get_grad_phi(self):
        host = 1
        dphi_dx, dphi_dy = self.unstructured_grid.get_grad_phi_wrapper(host)
        test.assert_array_almost_equal(dphi_dx, [1., -1., 0.])
        test.assert_array_almost_equal(dphi_dy, [-1., 0., 1.])

    def test_shephard_interpolation(self):
        xpts = np.array([-2.0, -1.0, 1.0, 2.0], dtype=DTYPE_FLOAT)
        ypts = np.array([-2.0, -1.0, 1.0, 2.0], dtype=DTYPE_FLOAT)
        vals = np.array([0.0, 0.0, 1.0, 0.0], dtype=DTYPE_FLOAT)

        val = self.unstructured_grid.shepard_interpolation_wrapper(1.0, 1.0, xpts, ypts, vals)
        test.assert_almost_equal(val, 1.0)

        val = self.unstructured_grid.shepard_interpolation_wrapper(0.0, 0.0, xpts, ypts, vals)
        test.assert_almost_equal(val, 0.4)


class UnstructuredGeographicGrid_test(TestCase):
    """ Unit tests for unstructured Geographic grids

    The tests use the same grid as is used with the Arakawa A data reader tests. Some
    of the tests are also the same, but here they don't go through Arakawa A data reader.
    """

    def setUp(self):
        # Conversion factor
        self.deg_to_radians = np.radians(1)

        # Basic grid (4 x 3 x 4).
        latitude = np.array([11., 12., 13., 14], dtype=DTYPE_FLOAT)
        longitude = np.array([1., 2., 3.], dtype=DTYPE_FLOAT)

        # Mask [depth, lat, lon]. 1 is sea, 0 land. Note the last depth level is masked everywhere.
        mask = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=DTYPE_INT)

        # Switch the mask convention to that which PyLag anticipates. i.e. 1 is a masked point, 0 a non-masked point.
        mask = 1 - mask

        # Separately save the surface mask at nodes. This is taken as the land sea mask.
        land_sea_mask = mask[:, :]
        land_sea_mask_nodes = np.moveaxis(land_sea_mask, 0, 1)  # Move to [lon, lat]
        land_sea_mask_nodes = land_sea_mask_nodes.reshape(np.prod(land_sea_mask_nodes.shape), order='C')

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
        nv = np.asarray(np.flip(tri.simplices.copy(), axis=1), dtype=DTYPE_INT)
        n_elements = nv.shape[0]

        # Save neighbours
        #   - Transpose to give it the dimension ordering expected by PyLag
        #   - Sort to ensure match with nv
        nbe = np.asarray(tri.neighbors, dtype=DTYPE_INT)
        gm.sort_adjacency_array(nv, nbe)

        # Save lon and lat points at element centres
        lon_elements, lat_elements = gm.compute_element_midpoints_in_geographic_coordinates(nv, lon_nodes, lat_nodes)

        # Convert to radians
        lon_nodes = lon_nodes * self.deg_to_radians
        lat_nodes = lat_nodes * self.deg_to_radians
        lon_elements = lon_elements * self.deg_to_radians
        lat_elements = lat_elements * self.deg_to_radians

        # Generate the land-sea mask for elements
        land_sea_mask_elements = np.empty(n_elements, dtype=DTYPE_INT)
        gm.compute_land_sea_element_mask(nv, land_sea_mask_nodes, land_sea_mask_elements, 2)

        # Transpose arrays
        nv = nv.T
        nbe = nbe.T

        # Flag open boundaries with -2 flag
        nbe[np.asarray(nbe == -1).nonzero()] = -2

        # Save grid variables
        self.name = b'test_grid'
        self.n_elems = n_elements
        self.n_nodes = n_nodes
        self.nv = np.ascontiguousarray(nv.astype(DTYPE_INT))
        self.nbe = np.ascontiguousarray(nbe.astype(DTYPE_INT))
        self.x = np.ascontiguousarray(lon_nodes.astype(DTYPE_FLOAT))
        self.y = np.ascontiguousarray(lat_nodes.astype(DTYPE_FLOAT))
        self.xc = np.ascontiguousarray(lon_elements.astype(DTYPE_FLOAT))
        self.yc = np.ascontiguousarray(lat_elements.astype(DTYPE_FLOAT))
        self.land_sea_mask_elements = np.ascontiguousarray(land_sea_mask_elements.astype(DTYPE_INT))
        self.land_sea_mask_nodes = np.ascontiguousarray(land_sea_mask_nodes.astype(DTYPE_INT))

        # Create config
        config = configparser.ConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')

        # Create unstructured grid
        self.unstructured_grid = UnstructuredGeographicGrid(config, self.name, self.n_nodes, self.n_elems,
                                                            self.nv, self.nbe, self.x, self.y, self.xc,
                                                            self.yc, self.land_sea_mask_elements,
                                                            self.land_sea_mask_nodes)

    def tearDown(self):
        del self.unstructured_grid

    def test_find_host_using_global_search(self):
        particle = ParticleSmartPtr(x1 = self.deg_to_radians * 1.666666667, x2 = self.deg_to_radians * 11.666666667)
        flag = self.unstructured_grid.find_host_using_global_search_wrapper(particle)
        test.assert_equal(particle.get_host_horizontal_elem('test_grid'), 0)
        test.assert_equal(flag, 0)

    def test_find_host_when_particle_is_in_the_domain(self):
        particle_new = ParticleSmartPtr(x1 = self.deg_to_radians * 2.333333333, x2 = self.deg_to_radians * 11.6666666667,
                                        host_elements={'test_grid':  7})
        flag = self.unstructured_grid.find_host_using_local_search_wrapper(particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.get_host_horizontal_elem('test_grid'), 6)

    def test_get_phi_when_particle_is_at_an_elements_vertex(self):
        # Vertex 0
        x1 = self.deg_to_radians * 2.0
        x2 = self.deg_to_radians * 12.0
        host = 0
        phi = self.unstructured_grid.get_phi(x1, x2, host)
        test.assert_array_almost_equal(phi, [1., 0., 0.])

        # Vertex 1
        x1 = self.deg_to_radians * 2.0
        x2 = self.deg_to_radians * 11.0
        host = 0
        phi = self.unstructured_grid.get_phi(x1, x2, host)
        test.assert_array_almost_equal(phi, [0., 1., 0.])

        # Vertex 2
        x1 = self.deg_to_radians * 1.0
        x2 = self.deg_to_radians * 12.0
        host = 0
        phi = self.unstructured_grid.get_phi(x1, x2, host)
        test.assert_array_almost_equal(phi, [0., 0., 1.])

    def test_get_boundary_intersection(self):
        particle_old = ParticleSmartPtr(x1 = self.deg_to_radians * 1.99, x2 = self.deg_to_radians * 12.9, host_elements={'test_grid': 5})
        particle_new = ParticleSmartPtr(x1 = self.deg_to_radians * 1.99, x2 = self.deg_to_radians * 13.1, host_elements={'test_grid': 5})

        start_point, end_point, intersection = self.unstructured_grid.get_boundary_intersection_wrapper(particle_old, particle_new)

        test.assert_almost_equal(start_point[0], np.radians(1.0))
        test.assert_almost_equal(start_point[1], np.radians(13.0))
        test.assert_almost_equal(end_point[0], np.radians(2.0))
        test.assert_almost_equal(end_point[1], np.radians(13.0))
        test.assert_almost_equal(intersection[0], np.radians(1.99), decimal=4)
        test.assert_almost_equal(intersection[1], np.radians(13.0), decimal=4)

    def test_interpolate_in_space(self):
        h_grid = np.array([25., 10., 999., 999.,  25.,  10., 999., 999.,  25.,  10., 999., 999.], dtype=DTYPE_FLOAT)

        # Set the particle at the element's centroid
        x1 = self.deg_to_radians * 1.6670652236426569
        x2 = self.deg_to_radians * 11.667054258869966
        host_elements = {'test_grid': 0}

        particle = ParticleSmartPtr(x1=x1, x2=x2, host_elements=host_elements)
        self.unstructured_grid.set_local_coordinates_wrapper(particle)
        h = self.unstructured_grid.interpolate_in_space_wrapper(h_grid, particle)
        test.assert_almost_equal(h, 15.)

    def test_interpolate_in_space_at_the_centre_of_a_boundary_element(self):
        h_grid = np.array([25., 10., 999., 999.,  25.,  10., 999., 999.,  25.,  10., 999., 999.], dtype=DTYPE_FLOAT)

        # Set the particle at the element's centroid
        x1 = self.deg_to_radians * 1.667100645520906
        x2 = self.deg_to_radians * 12.66708505715026
        host_elements = {'test_grid': 5}

        particle = ParticleSmartPtr(x1=x1, x2=x2, host_elements=host_elements)
        self.unstructured_grid.set_local_coordinates_wrapper(particle)
        h = self.unstructured_grid.interpolate_in_space_wrapper(h_grid, particle)
        test.assert_almost_equal(h, 10.)

    def test_interpolate_in_space_on_a_boundary_element(self):
        h_grid = np.array([25., 10., 999., 999.,  25.,  10., 999., 999.,  25.,  10., 999., 999.], dtype=DTYPE_FLOAT)

        # Set the particle at the element's centroid
        x1 = self.deg_to_radians * 2.
        x2 = self.deg_to_radians * 13.
        host_elements = {'test_grid': 5}

        particle = ParticleSmartPtr(x1=x1, x2=x2, host_elements=host_elements)
        self.unstructured_grid.set_local_coordinates_wrapper(particle)
        h = self.unstructured_grid.interpolate_in_space_wrapper(h_grid, particle)
        test.assert_almost_equal(h, 10.)

