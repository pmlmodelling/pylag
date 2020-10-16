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
        self.nv = np.array([[4, 0, 2, 4, 3], [3, 1, 4, 5, 6], [1, 3, 1, 3, 0]], dtype=DTYPE_INT)
        self.nbe = np.array([[1, 0, 0, -1, -1], [2, 4, -1, 0, 1], [3, -1, -1, -1, -1]], dtype=DTYPE_INT)
        self.x = np.array([2.0, 1.0, 0.0, 2.0, 1.0, 1.5, 3.0], dtype=DTYPE_FLOAT)
        self.y = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0], dtype=DTYPE_FLOAT)
        self.xc = np.array([1.3333333333, 1.6666666667, 0.6666666667, 1.5000000000, 2.3333333333], dtype=DTYPE_FLOAT)
        self.yc = np.array([1.6666666667, 1.3333333333, 1.3333333333, 2.3333333333, 1.3333333333], dtype=DTYPE_FLOAT)

        # Create config
        config = configparser.ConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')

        # Create unstructured grid
        self.unstructured_grid = UnstructuredCartesianGrid(config, self.name, self.n_nodes, self.n_elems,
                                                           self.nv, self.nbe, self.x, self.y, self.xc,
                                                           self.yc)
        
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
        intersection = self.unstructured_grid.get_boundary_intersection_wrapper(particle_old, particle_new)
        test.assert_almost_equal(intersection.x1_py, 2.0)
        test.assert_almost_equal(intersection.y1_py, 1.0)
        test.assert_almost_equal(intersection.x2_py, 1.0)
        test.assert_almost_equal(intersection.y2_py, 1.0)
        test.assert_almost_equal(intersection.xi_py, 1.6666666667)
        test.assert_almost_equal(intersection.yi_py, 1.0)

    def test_get_boundary_intersection_when_a_particle_has_left_an_external_elements_edge(self):
        particle_old = ParticleSmartPtr(x1=1.5, x2=1.0, host_elements={'test_grid': 1})
        particle_new = ParticleSmartPtr(x1=1.5, x2=0.9, host_elements={'test_grid': 1})
        intersection = self.unstructured_grid.get_boundary_intersection_wrapper(particle_old, particle_new)
        test.assert_almost_equal(intersection.x1_py, 2.0)
        test.assert_almost_equal(intersection.y1_py, 1.0)
        test.assert_almost_equal(intersection.x2_py, 1.0)
        test.assert_almost_equal(intersection.y2_py, 1.0)
        test.assert_almost_equal(intersection.xi_py, 1.5)
        test.assert_almost_equal(intersection.yi_py, 1.0)

    def test_get_boundary_intersection_when_a_particle_has_moved_to_an_external_elements_edge(self):
        particle_old = ParticleSmartPtr(x1=1.5, x2=1.1, host_elements={'test_grid': 1})
        particle_new = ParticleSmartPtr(x1=1.5, x2=1.0, host_elements={'test_grid': 1})
        intersection = self.unstructured_grid.get_boundary_intersection_wrapper(particle_old, particle_new)
        test.assert_almost_equal(intersection.x1_py, 2.0)
        test.assert_almost_equal(intersection.y1_py, 1.0)
        test.assert_almost_equal(intersection.x2_py, 1.0)
        test.assert_almost_equal(intersection.y2_py, 1.0)
        test.assert_almost_equal(intersection.xi_py, 1.5)
        test.assert_almost_equal(intersection.yi_py, 1.0)

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

        val = self.unstructured_grid.shepard_interpolation(1.0, 1.0, xpts, ypts, vals)
        test.assert_almost_equal(val, 1.0)

        val = self.unstructured_grid.shepard_interpolation(0.0, 0.0, xpts, ypts, vals)
        test.assert_almost_equal(val, 0.4)


class UnstructuredGeographicGrid_test(TestCase):
    """ Unit tests for unstructured Geographic grids

    The tests use the same grid as is used with the Arakawa A data reader tests. Some
    of the tests are also the same, but here they don't go through Arakawa A data reader.
    """

    def setUp(self):
        # Basic grid (3 x 3 x 4).
        latitude = np.array([11., 12., 13.], dtype=DTYPE_FLOAT)
        longitude = np.array([1., 2., 3.], dtype=DTYPE_FLOAT)

        # Mask [depth, lat, lon]. 1 is sea, 0 land. Note the last depth level is masked everywhere.
        mask = np.array([[[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                         [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                         [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=DTYPE_INT)

        # Switch the mask convention to that which PyLag anticipates. i.e. 1 is a masked point, 0 a non-masked point.
        mask = 1 - mask

        # Separately save the surface mask at nodes. This is taken as the land sea mask.
        land_sea_mask = mask[0, :, :]
        land_sea_mask_nodes = np.moveaxis(land_sea_mask, 0, 1)  # Move to [lon, lat]
        land_sea_mask_nodes = land_sea_mask_nodes.reshape(np.prod(land_sea_mask_nodes.shape), order='C')

        # Mask one extra point (node 1) to help with testing wet dry status calls
        mask[:, 1, 0] = 1

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
        lon_elements = np.empty(n_elements, dtype=DTYPE_FLOAT)
        lat_elements = np.empty(n_elements, dtype=DTYPE_FLOAT)
        for i, element in enumerate(range(n_elements)):
            lon_elements[i] = lon_nodes[(nv[element, :])].mean()
            lat_elements[i] = lat_nodes[(nv[element, :])].mean()

        # Generate the land-sea mask for elements
        land_sea_mask_elements = np.empty(n_elements, dtype=DTYPE_INT)
        gm.compute_land_sea_element_mask(nv, land_sea_mask_nodes, land_sea_mask_elements)

        # Transpose arrays
        nv = nv.T
        nbe = nbe.T

        # Flag open boundaries with -2 flag
        nbe[np.asarray(nbe == -1).nonzero()] = -2

        # Flag land boundaries with -1 flag
        land_elements = np.asarray(land_sea_mask_elements == 1).nonzero()[0]
        for element in land_elements:
            nbe[np.asarray(nbe == element).nonzero()] = -1

        # Save grid variables
        self.name = b'test_grid'
        self.n_elems = n_elements
        self.n_nodes = n_nodes
        self.nv = nv.astype(DTYPE_INT)
        self.nbe = nbe.astype(DTYPE_INT)
        self.x = lon_nodes.astype(DTYPE_FLOAT)
        self.y = lat_nodes.astype(DTYPE_FLOAT)
        self.xc = lon_elements.astype(DTYPE_FLOAT)
        self.yc = lat_elements.astype(DTYPE_FLOAT)

        # Create config
        config = configparser.ConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')

        # Create unstructured grid
        self.unstructured_grid = UnstructuredGeographicGrid(config, self.name, self.n_nodes, self.n_elems,
                                                            self.nv, self.nbe, self.x, self.y, self.xc,
                                                            self.yc)

    def tearDown(self):
        del self.unstructured_grid

    def test_find_host_using_global_search(self):
        particle = ParticleSmartPtr(x1=1.666666667, x2=11.666666667)
        flag = self.unstructured_grid.find_host_using_global_search_wrapper(particle)
        test.assert_equal(particle.get_host_horizontal_elem('test_grid'), 0)
        test.assert_equal(flag, 0)

    def test_find_host_when_particle_is_in_the_domain(self):
        particle_new = ParticleSmartPtr(x1=2.333333333, x2=11.6666666667,
                                        host_elements={'test_grid':  5})
        flag = self.unstructured_grid.find_host_using_local_search_wrapper(particle_new)
        test.assert_equal(flag, 0)
        test.assert_equal(particle_new.get_host_horizontal_elem('test_grid'), 4)

    def test_get_phi_when_particle_is_at_an_elements_vertex(self):
        # Vertex 0
        x1 = 2.0
        x2 = 12.0
        host = 0
        phi = self.unstructured_grid.get_phi(x1, x2, host)
        test.assert_array_almost_equal(phi, [1., 0., 0.])

        # Vertex 1
        x1 = 2.0
        x2 = 11.0
        host = 0
        phi = self.unstructured_grid.get_phi(x1, x2, host)
        test.assert_array_almost_equal(phi, [0., 1., 0.])

        # Vertex 2
        x1 = 1.0
        x2 = 12.0
        host = 0
        phi = self.unstructured_grid.get_phi(x1, x2, host)
        test.assert_array_almost_equal(phi, [0., 0., 1.])

