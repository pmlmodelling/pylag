from unittest import TestCase
import numpy as np
import numpy.testing as test

from pylag.data_types_python import DTYPE_FLOAT

from pylag.math import cartesian_to_sigma_coords, sigma_to_cartesian_coords
from pylag.cwrappers import det_wrapper, inner_product_wrapper, get_intersection_point_wrapper

def test_det():
    a = [1.0, 2.0]
    b = [3.0, 4.0]
    c = det_wrapper(a, b)
    test.assert_array_almost_equal(c, -2.0)

def test_inner_product():
    a = [1.0, 2.0]
    b = [3.0, 4.0]
    c = inner_product_wrapper(a, b)
    test.assert_array_almost_equal(c, 11.0)

def test_get_intersection_point_for_perpendicular_lines():
    x1 = np.array([0.0, 2.0], dtype=DTYPE_FLOAT)
    x2 = np.array([0.0, 0.0], dtype=DTYPE_FLOAT)
    x3 = np.array([2.0, 1.0], dtype=DTYPE_FLOAT)
    x4 = np.array([-1.0, 1.0], dtype=DTYPE_FLOAT)
    xi = np.empty([2], dtype=DTYPE_FLOAT)    
    get_intersection_point_wrapper(x1, x2, x3, x4, xi)
    test.assert_array_almost_equal(xi, [0.0, 1.0])

def test_get_intersection_point_for_angled_lines():
    x1 = np.array([-1.0, -1.0], dtype=DTYPE_FLOAT)
    x2 = np.array([1.0, 1.0], dtype=DTYPE_FLOAT)
    x3 = np.array([0.0, -1.0], dtype=DTYPE_FLOAT)
    x4 = np.array([0.0, 1.0], dtype=DTYPE_FLOAT)
    xi = np.empty([2], dtype=DTYPE_FLOAT)    
    get_intersection_point_wrapper(x1, x2, x3, x4, xi)
    test.assert_array_almost_equal(xi, [0.0, 0.0])

def test_cartesian_to_sigma_coords():
    h = -50.0
    zeta = 2.0

    z = 2.0
    sigma = cartesian_to_sigma_coords(z, h, zeta)
    test.assert_almost_equal(sigma, 0.0)

    z = -24.0
    sigma = cartesian_to_sigma_coords(z, h, zeta)
    test.assert_almost_equal(sigma, -0.5)

    z = -50.0
    sigma = cartesian_to_sigma_coords(z, h, zeta)
    test.assert_almost_equal(sigma, -1.0)   
