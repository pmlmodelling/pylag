from unittest import TestCase
import numpy.testing as test
import numpy as np

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.particle_cpp_wrapper import ParticleSmartPtr
from pylag.unstructured import UnstructuredCartesianGrid


class UnstructuredCartesianGrid_test(TestCase):

    def setUp(self):
        self.name = b'test_grid'
        self.n_elems = 5
        self.n_nodes = 7
        self.nv = np.array([[4, 0, 2, 4, 3], [3, 1, 4, 5, 6], [1, 3, 1, 3, 0]], dtype=int)
        self.nbe = np.array([[1, 0, 0, -1, -1], [2, 4, -1, 0, 1], [3, -1, -1, -1, -1]], dtype=int)
        self.x = np.array([2.0, 1.0, 0.0, 2.0, 1.0, 1.5, 3.0], dtype=float)
        self.y = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0], dtype=float)
        self.xc = np.array([1.3333333333, 1.6666666667, 0.6666666667, 1.5000000000, 2.3333333333], dtype=float)
        self.yc = np.array([1.6666666667, 1.3333333333, 1.3333333333, 2.3333333333, 1.3333333333], dtype=float)

        # Create config
        config = configparser.ConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')

        # Create data reader
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
        phi_min = np.min(np.array(particle.get_phi('test_grid'), dtype=float))
        test.assert_equal(np.abs(phi_min), 0.0)

    #def test_get_phi_when_particle_is_at_an_elements_centre(self):
    #    x1 = 1.666666667
    #    x2 = 1.333333333
    #    host = 1
    #    phi = np.array([-999., -999., -999.])
    #    self.unstructured_grid.get_phi(x1, x2, host, phi)
    #    test.assert_array_almost_equal(phi, [0.5, 0.5, 0.5])

